//
//  ModelDerivation.swift
//  DrewsChessMachine
//
//  The engine behind `--derive-model`: make a new model file from an existing
//  one by applying a list of *derive operations* — architecture edits that
//  never change a tensor's shape, each of which re-initializes only the
//  tensors it declares. Everything else is copied bit-exact from the source
//  file. The motivating case is a paired copy for an A/B: the same fresh net
//  with one init choice flipped (GitHub issue #7, `se_beta_init`), so the two
//  arms differ in exactly that and nothing else.
//
//  Structure — built to grow:
//  - `DeriveOperationKind` is one operation's catalog entry: its CLI flag,
//    value syntax, the architecture fields it changes, the tensors it
//    rewrites, and a factory. `ModelDerivation.operationKinds` is the ONE
//    list of kinds; the CLI parser, `--derive-model --help`, and the docs
//    are all driven from it. Adding an operation = one new `DeriveOperation`
//    type + one entry in that list.
//  - `DeriveOperation` is one configured operation: it validates itself
//    against the source architecture, returns the target architecture, and
//    lists the tensor rewrites (name, rewritten element ranges, rewrite).
//  - `ModelDerivation.derive` applies the operations and enforces the
//    guardrails that do not depend on which operation ran:
//      * the source must be a plain model file (no optimizer velocity, no
//        trainer schedule — exact-resume state is meaningless once weights
//        are re-initialized);
//      * an operation that rewrites tensors needs an untrained source,
//        because the rewrite would reset learned weights while the derived
//        file still claims the source's training step and lineage. Any
//        positive evidence of training refuses: a recorded `training_step`
//        above zero, a lineage step total above zero, a lineage parent that
//        stated a positive trainer step, or an earlier graft's positive
//        `source_training_step` (a malformed recorded step is refused too,
//        never read as "absent"). A file written before lineage that was
//        trained but states none of these cannot be told from a fresh one.
//        Operations that rewrite no tensor stay allowed on a trained source;
//      * the target architecture must validate and must have exactly the
//        source's tensor plan (names AND shapes) — a shape-changing request
//        is refused before anything is written;
//      * every element outside an operation's declared ranges is verified
//        bit-identical to the source after the rewrite, so an operation that
//        writes outside what it declares is caught, not shipped;
//      * the output is decoded back through the normal loader before it is
//        returned.
//  - Lineage: a new ModelID, `parent_model_id` = the source's ModelID, a
//    human `notes` line, and `derivation_history` — a JSON array carrying
//    the source's own history (if it was itself derived) plus one record for
//    this derivation listing every operation applied, the fields it changed,
//    the tensors it rewrote, and the source file's SHA-256. A chain of
//    derivations therefore stays traceable from the newest file alone.
//
//  Every other `__metadata__` key of the source (training step, value-head
//  centering marker, replay provenance, …) is carried over verbatim; the
//  keys this file rewrites are listed in `rewrittenMetadataKeys`.
//

import CryptoKit
import Foundation

// MARK: - Operation protocol + catalog entry

/// One tensor an operation rewrites: which elements (in the ON-DISK PyTorch
/// layout the safetensors file stores) it may change, and how.
struct DeriveTensorRewrite: Sendable {
    let tensorName: String
    /// Element ranges of the flat on-disk data the rewrite may change. The
    /// engine verifies every element outside them is bit-identical afterward.
    let rewrittenElementRanges: [Range<Int>]
    /// Short human description for logs and the derivation record.
    let summary: String
    /// Rewrites the tensor's flat on-disk data in place.
    let rewrite: @Sendable (inout [Float]) -> Void
}

/// A configured derive operation. See `ModelDerivation` for the contract.
protocol DeriveOperation: Sendable {
    /// The `DeriveOperationKind.name` this operation is an instance of.
    var kindName: String { get }
    /// The arguments as they should appear in the derivation record.
    var recordedArguments: [String: String] { get }
    /// Validate against `architecture` and return the edited architecture.
    /// Throws when the request does not apply (wrong SE style, nothing to
    /// change, group out of range, …).
    func apply(to architecture: NetworkArchitecture) throws -> NetworkArchitecture
    /// The tensor rewrites turning `source`-architecture weights into
    /// `target`-architecture weights (target = `apply(to: source)`).
    func tensorRewrites(source: NetworkArchitecture, target: NetworkArchitecture) throws -> [DeriveTensorRewrite]
}

/// The catalog entry for one kind of derive operation — what the CLI parses,
/// what `--help` lists, and what the derivation record names.
struct DeriveOperationKind: Sendable {
    /// Stable name, recorded in `derivation_history` (e.g. `set-se-beta-init`).
    let name: String
    /// The CLI flag that requests it (e.g. `--set-se-beta-init`).
    let flag: String
    /// The flag's value syntax for `--help` (e.g. `glorot|zero`).
    let valueSyntax: String
    /// One-paragraph description for `--help` and the docs.
    let summary: String
    /// The architecture JSON fields this kind may change.
    let changedArchitectureFields: [String]
    /// Human description of the tensors this kind rewrites.
    let rewrittenTensorsDescription: String
    /// Whether `--group <index>` (repeatable) narrows it to chosen block groups.
    let acceptsGroupSelection: Bool
    /// Builds the operation from the flag's value and the `--group` indices
    /// (nil when none were given). Throws on an unparseable value.
    let make: @Sendable (_ value: String, _ groupIndices: [Int]?) throws -> any DeriveOperation
}

// MARK: - Engine

enum ModelDerivation {

    /// Every derive operation this build supports — the single place to add
    /// one. The CLI flags, `--help`, and the docs all come from this list.
    static let operationKinds: [DeriveOperationKind] = [
        SetSEBetaInitDeriveOperation.kind,
        SetActivationDeriveOperation.kind,
        SetSEActivationDeriveOperation.kind,
    ] + SetSiteActivationDeriveOperation.kinds + [
        SetRezeroAlphaInitDeriveOperation.kind,
        SetRezeroAlphaCapDeriveOperation.kind,
        SetNeutralInitDeriveOperation.kind,
        SetSEGammaBiasInitDeriveOperation.kind,
        SetBranchOutputInitDeriveOperation.kind,
        SetSkipProjectionInitDeriveOperation.kind,
        SetPolicyHeadFinalInitDeriveOperation.kind,
        SetValueHeadFinalInitDeriveOperation.kind,
        SetValueHeadDrawPriorDeriveOperation.kind,
    ]

    /// The kind whose `flag` is `flag`, if any.
    static func kind(forFlag flag: String) -> DeriveOperationKind? {
        operationKinds.first { $0.flag == flag }
    }

    /// `__metadata__` key holding the JSON array of `DerivationRecord`s.
    static let derivationHistoryKey = "derivation_history"
    /// `creator` value stamped on derived files.
    static let creator = "derive-model"

    /// Source `__metadata__` keys this derivation replaces rather than copies.
    static let rewrittenMetadataKeys: Set<String> = Set([
        SafetensorsModelIO.Key.formatVersion,
        SafetensorsModelIO.Key.modelID,
        SafetensorsModelIO.Key.createdAt,
        SafetensorsModelIO.Key.creator,
        SafetensorsModelIO.Key.parentModelID,
        SafetensorsModelIO.Key.notes,
        SafetensorsModelIO.Key.architecture,
        SafetensorsFile.contentHashKey,
        derivationHistoryKey,
        LineageRecord.metadataKey,
    ]).union(LineageRecord.MirrorKey.all)

    enum DeriveError: Error, CustomStringConvertible, Equatable {
        case noOperations
        case sourceHasOptimizerState(source: String, detail: String)
        case sourceMissingModelID(source: String)
        case operationNotApplicable(operation: String, detail: String)
        case invalidTargetArchitecture(detail: String)
        case shapeChangingRequest(detail: String)
        case missingTensor(name: String)
        case rewriteChangedElementCount(name: String, before: Int, after: Int)
        case rewriteOutsideDeclaredRanges(name: String, elementIndex: Int)
        case unreadableDerivationHistory(detail: String)
        case outputFailedVerification(detail: String)
        case sourceIsTrained(source: String, trainingStep: Int)
        case sourceIsTrainedByLineage(source: String, evidence: String)
        case malformedSourceTrainingStep(source: String, key: String, value: String)

        var description: String {
            switch self {
            case .noOperations:
                return "no derive operation requested"
            case .sourceHasOptimizerState(let source, let detail):
                return "\(source) carries optimizer/exact-resume state (\(detail)); derive from a model file "
                    + "(a fresh net or champion), not a trainer-state file — re-initialized weights would "
                    + "not match the saved optimizer state"
            case .sourceMissingModelID(let source):
                return "\(source) has no model_id, so the derived file could not record its parent"
            case .operationNotApplicable(let operation, let detail):
                return "\(operation): \(detail)"
            case .invalidTargetArchitecture(let detail):
                return "the derived architecture does not validate: \(detail)"
            case .shapeChangingRequest(let detail):
                return "refused: the requested change would alter the tensor layout (\(detail)); "
                    + "--derive-model only makes changes that keep every tensor's name and shape"
            case .missingTensor(let name):
                return "source file has no tensor '\(name)'"
            case .rewriteChangedElementCount(let name, let before, let after):
                return "internal error: rewrite of '\(name)' changed its element count \(before) -> \(after)"
            case .rewriteOutsideDeclaredRanges(let name, let index):
                return "internal error: rewrite of '\(name)' changed element \(index), outside its declared ranges"
            case .unreadableDerivationHistory(let detail):
                return "source \(ModelDerivation.derivationHistoryKey) is unreadable: \(detail)"
            case .outputFailedVerification(let detail):
                return "the derived file failed to load back: \(detail)"
            case .sourceIsTrained(let source, let trainingStep):
                return "\(source) records training_step \(trainingStep); a derive operation that rewrites tensors "
                    + "would reset learned weights while the derived file kept that training step and lineage — "
                    + "derive tensor-rewriting variants from a fresh (untrained) net"
            case .sourceIsTrainedByLineage(let source, let evidence):
                return "\(source) holds trained weights (\(evidence)) although it records no positive training_step; "
                    + "a derive operation that rewrites tensors would reset learned weights while the derived file "
                    + "kept that lineage — derive tensor-rewriting variants from a fresh (untrained) net"
            case .malformedSourceTrainingStep(let source, let key, let value):
                return "\(source) records \(key) '\(value)', which is not a non-negative integer; "
                    + "refusing to rewrite tensors without knowing whether the source is trained"
            }
        }
    }

    /// One operation as recorded in a `DerivationRecord`.
    ///
    /// The fields after `rewrittenTensors` are written only by a graft
    /// (`--graft-to`); every other operation leaves them nil, and nil fields
    /// are omitted when encoded, so the records of same-layout operations —
    /// and every history written before grafts existed — read and write
    /// exactly as before.
    struct OperationRecord: Codable, Sendable, Equatable {
        let operation: String
        let arguments: [String: String]
        let changedArchitectureFields: [String]
        /// Tensors the operation wrote. For a graft: every target tensor it
        /// initialized (all of them freshly drawn or builder constants).
        let rewrittenTensors: [String]
        /// Graft: target tensors copied bit-exact from the source, as
        /// `target name` (or `source name -> target name` when renamed).
        var copiedTensors: [String]? = nil
        /// Graft: source tensors with no place in the target.
        var droppedTensors: [String]? = nil
        /// Graft: the init seed the initialized tensors were drawn under.
        var initSeed: String? = nil
        /// Graft: the weight-initialization scheme (`WeightInitScheme.current`).
        var initRuleVersion: String? = nil
        /// Graft: how each initialized tensor got its values —
        /// `RandomTensorRole` for a drawn tensor, `builder_constant` for one
        /// the graph builder sets to a fixed value.
        var perTensorInit: [String: String]? = nil

        init(operation: String, arguments: [String: String], changedArchitectureFields: [String],
             rewrittenTensors: [String]) {
            self.operation = operation
            self.arguments = arguments
            self.changedArchitectureFields = changedArchitectureFields
            self.rewrittenTensors = rewrittenTensors
        }

        enum CodingKeys: String, CodingKey {
            case operation
            case arguments
            case changedArchitectureFields = "changed_architecture_fields"
            case rewrittenTensors = "rewritten_tensors"
            case copiedTensors = "copied_tensors"
            case droppedTensors = "dropped_tensors"
            case initSeed = "init_seed"
            case initRuleVersion = "init_rule_version"
            case perTensorInit = "per_tensor_init"
        }
    }

    /// One derivation step, as stored in `derivation_history`.
    struct DerivationRecord: Codable, Sendable, Equatable {
        let modelID: String
        let parentModelID: String
        let sourceFile: String
        let sourceSHA256: String
        let sourceFormatVersion: Int
        let createdAtUnix: Int64
        let build: String
        let operations: [OperationRecord]

        enum CodingKeys: String, CodingKey {
            case modelID = "model_id"
            case parentModelID = "parent_model_id"
            case sourceFile = "source_file"
            case sourceSHA256 = "source_sha256"
            case sourceFormatVersion = "source_format_version"
            case createdAtUnix = "created_at_unix"
            case build
            case operations
        }
    }

    /// What `derive` produced.
    struct Result: Sendable {
        /// The derived safetensors bytes, ready to write.
        let data: Data
        let sourceArchitecture: NetworkArchitecture
        let targetArchitecture: NetworkArchitecture
        /// This derivation's record (also the last entry of `history`).
        let record: DerivationRecord
        /// The full `derivation_history` written to the file.
        let history: [DerivationRecord]
        /// The derived file's lineage record.
        let lineage: LineageRecord
        /// Every rewrite applied, in order, for logging.
        let rewrites: [(operation: String, tensorName: String, summary: String)]
        /// The source's embedded-architecture decode (for legacy logging).
        let sourceArchitectureFormat: ArchitectureFormat.DecodeFormat
    }

    /// Apply `operations` to the model in `sourceData` (read from
    /// `sourceName`). See the file header for the guardrails. Pure: no file
    /// or log I/O — the caller writes `Result.data` and logs.
    static func derive(
        sourceData: Data,
        sourceName: String,
        operations: [any DeriveOperation],
        newModelID: String,
        createdAtUnix: Int64,
        build: String,
        invocationArguments: [String]
    ) throws -> Result {
        guard !operations.isEmpty else { throw DeriveError.noOperations }

        let (tensors, sourceMetadata) = try SafetensorsFile.decode(sourceData)
        let parentModelID = try requirePlainModelSource(tensors: tensors, metadata: sourceMetadata, sourceName: sourceName)

        // The full loader must accept the source (plan, dims, value-head
        // marker) — the derived file is only as loadable as its source.
        let sourceDecoded = try SafetensorsModelIO.decode(sourceData, valueHead: .asStored, source: sourceName)
        let source = sourceDecoded.architecture

        var target = source
        for operation in operations {
            target = try operation.apply(to: target)
        }
        do {
            try target.validate()
        } catch {
            throw DeriveError.invalidTargetArchitecture(detail: String(describing: error))
        }
        try requireSameTensorLayout(source: source, target: target)

        // Rewrite. Each operation's rewrites are computed against the
        // architecture it was applied to, in order.
        var byName: [String: Int] = [:]
        for (index, tensor) in tensors.enumerated() { byName[tensor.name] = index }
        var outputTensors = tensors
        var appliedRewrites: [(operation: String, tensorName: String, summary: String)] = []
        var operationRecords: [OperationRecord] = []
        var stepSource = source
        for operation in operations {
            let stepTarget = try operation.apply(to: stepSource)
            let rewrites = try operation.tensorRewrites(source: stepSource, target: stepTarget)
            for rewrite in rewrites {
                guard let index = byName[rewrite.tensorName] else {
                    throw DeriveError.missingTensor(name: rewrite.tensorName)
                }
                let before = outputTensors[index]
                var data = before.data
                rewrite.rewrite(&data)
                guard data.count == before.data.count else {
                    throw DeriveError.rewriteChangedElementCount(
                        name: rewrite.tensorName, before: before.data.count, after: data.count)
                }
                try requireUnchangedOutside(
                    rewrite.rewrittenElementRanges, before: before.data, after: data, name: rewrite.tensorName)
                outputTensors[index] = SafetensorsTensor(name: before.name, shape: before.shape, data: data)
                appliedRewrites.append((operation.kindName, rewrite.tensorName, rewrite.summary))
            }
            guard let kind = operationKinds.first(where: { $0.name == operation.kindName }) else {
                preconditionFailure("derive operation '\(operation.kindName)' is not in ModelDerivation.operationKinds")
            }
            operationRecords.append(OperationRecord(
                operation: operation.kindName,
                arguments: operation.recordedArguments,
                changedArchitectureFields: kind.changedArchitectureFields,
                rewrittenTensors: rewrites.map(\.tensorName)))
            stepSource = stepTarget
        }

        // Lineage. The source as a parent carries its derivation history
        // (its lineage record's, or the flat key a file before lineage
        // states); this derivation appends one step to it.
        let sourceLineage = try SafetensorsModelIO.lineage(
            fromMetadata: sourceMetadata, formatVersion: sourceDecoded.architectureFormat.formatVersion)
        let sourceParent = LineageTracker.ParentFile(
            modelID: parentModelID,
            contentSHA256: sourceMetadata[SafetensorsFile.contentHashKey],
            trainerCompletedSteps: try SafetensorsModelIO.trainerClock(fromMetadata: sourceMetadata, source: sourceName),
            lineage: sourceLineage,
            derivationHistory: try LineageTracker.ParentFile.derivationHistory(lineage: sourceLineage, metadata: sourceMetadata))
        if !appliedRewrites.isEmpty {
            try requireUntrainedSource(sourceMetadata, lineage: sourceLineage,
                                       derivationHistory: sourceParent.derivationHistory, sourceName: sourceName)
        }
        let record = DerivationRecord(
            modelID: newModelID,
            parentModelID: parentModelID,
            sourceFile: sourceName,
            sourceSHA256: SHA256.hash(data: sourceData).map { String(format: "%02x", $0) }.joined(),
            sourceFormatVersion: sourceDecoded.architectureFormat.formatVersion,
            createdAtUnix: createdAtUnix,
            build: build,
            operations: operationRecords)

        var metadata = sourceMetadata.filter { !rewrittenMetadataKeys.contains($0.key) }
        metadata[SafetensorsModelIO.Key.formatVersion] = SafetensorsModelIO.formatVersion
        metadata[SafetensorsModelIO.Key.modelID] = newModelID
        metadata[SafetensorsModelIO.Key.createdAt] = String(createdAtUnix)
        metadata[SafetensorsModelIO.Key.creator] = creator
        metadata[SafetensorsModelIO.Key.parentModelID] = parentModelID
        metadata[SafetensorsModelIO.Key.notes] = notes(for: record)
        metadata[SafetensorsModelIO.Key.architecture] = String(decoding: try JSONEncoder().encode(target), as: UTF8.self)
        // The derived model's lineage: a new run whose weights carry the
        // source's history (totals continue from the source's record, or
        // stay unrecorded for a source written before lineage). Its
        // derivation history — the source's plus this step — is written
        // by the record, with the flat `derivation_history` key as its
        // mirror.
        let lineage = try LineageTracker.untrainedCopyRecord(
            source: sourceParent, derivation: record,
            sourceArchitecture: try LineageRecord.AncestorRun.ArchitectureAtDeparture.ifChanged(
                from: source, to: target, sourceMetadata: sourceMetadata,
                sourceFormatVersion: sourceDecoded.architectureFormat.formatVersion, sourceName: sourceName),
            pathKind: .derive, argv: invocationArguments,
            at: Date(timeIntervalSince1970: TimeInterval(createdAtUnix)))
        for (key, value) in try lineage.metadataEntries() { metadata[key] = value }

        let data = try SafetensorsFile.encode(tensors: outputTensors, metadata: metadata)

        // The derived file must load through the normal path and describe
        // exactly the target architecture.
        do {
            let reloaded = try SafetensorsModelIO.decode(data, valueHead: .asStored, source: "derived output")
            guard reloaded.architecture == target else {
                throw DeriveError.outputFailedVerification(detail: "embedded architecture differs from the target")
            }
        } catch let error as DeriveError {
            throw error
        } catch {
            throw DeriveError.outputFailedVerification(detail: String(describing: error))
        }

        return Result(
            data: data,
            sourceArchitecture: source,
            targetArchitecture: target,
            record: record,
            history: lineage.derivationHistory,
            lineage: lineage,
            rewrites: appliedRewrites,
            sourceArchitectureFormat: sourceDecoded.architectureFormat)
    }

    /// Read a file's `derivation_history`, or `[]` when it has none (it was
    /// not derived).
    static func decodeHistory(_ json: String?) throws -> [DerivationRecord] {
        guard let json else { return [] }
        do {
            return try JSONDecoder().decode([DerivationRecord].self, from: Data(json.utf8))
        } catch {
            throw DeriveError.unreadableDerivationHistory(detail: String(describing: error))
        }
    }

    /// The `notes` line for a derived file.
    static func notes(for record: DerivationRecord) -> String {
        let operations = record.operations.map { op -> String in
            let arguments = op.arguments.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: " ")
            return "\(op.operation) [\(arguments)] rewrote \(op.rewrittenTensors.count) tensors"
        }
        return "derived from \(record.parentModelID) (\(record.sourceFile), sha256 \(record.sourceSHA256)): "
            + operations.joined(separator: "; ")
    }

    /// Refuse a source that is not a plain model file — one carrying
    /// optimizer velocity or trainer schedule (exact-resume state means
    /// nothing once weights are re-initialized) — or that has no ModelID to
    /// record as the parent. Returns that ModelID.
    static func requirePlainModelSource(tensors: [SafetensorsTensor], metadata: [String: String],
                                        sourceName: String) throws -> String {
        if let velocity = tensors.first(where: { $0.name.hasPrefix("opt.") }) {
            throw DeriveError.sourceHasOptimizerState(source: sourceName, detail: "tensor '\(velocity.name)'")
        }
        if let scheduleKey = TrainerScheduleState.MetadataKey.all.first(where: { metadata[$0] != nil }) {
            throw DeriveError.sourceHasOptimizerState(source: sourceName, detail: "metadata '\(scheduleKey)'")
        }
        guard let parentModelID = metadata[SafetensorsModelIO.Key.modelID], !parentModelID.isEmpty else {
            throw DeriveError.sourceMissingModelID(source: sourceName)
        }
        return parentModelID
    }

    /// Refuse a tensor rewrite on a trained source. Any one piece of
    /// positive evidence of training refuses:
    /// - the raw `training_step` metadata is a positive integer (read raw so
    ///   a malformed value is an error, never mistaken for "no step");
    /// - the lineage record's step total (`cum_trainer_step`) is positive;
    /// - the record's parent stated a positive trainer step — the weights
    ///   descend from a trained file even when this file's own total is
    ///   unrecorded (a graft or champion copy of a file written before
    ///   lineage);
    /// - an earlier graft in the derivation history records a positive
    ///   `source_training_step` — a graft writes no `training_step`, and an
    ///   architecture-only derive after it keeps none either.
    ///
    /// A graft and a champion copy carry no `training_step` of their own,
    /// which is why the raw key alone is not enough. One gap remains: a file
    /// written before lineage that was trained but states no
    /// `training_step` and no derivation history carries no evidence at all,
    /// and is read as untrained.
    static func requireUntrainedSource(_ sourceMetadata: [String: String], lineage: LineageRecord.Presence,
                                       derivationHistory: [DerivationRecord], sourceName: String) throws {
        if let recorded = sourceMetadata[SafetensorsModelIO.Key.trainingStep] {
            guard let trainingStep = Int(recorded), trainingStep >= 0 else {
                throw DeriveError.malformedSourceTrainingStep(
                    source: sourceName, key: SafetensorsModelIO.Key.trainingStep, value: recorded)
            }
            if trainingStep > 0 {
                throw DeriveError.sourceIsTrained(source: sourceName, trainingStep: trainingStep)
            }
        }
        if let record = lineage.record {
            if let total = record.steps.cumTrainerStep, total > 0 {
                throw DeriveError.sourceIsTrainedByLineage(
                    source: sourceName, evidence: "its lineage records cum_trainer_step \(total)")
            }
            if let parent = record.parent, let parentStep = parent.trainerCompletedSteps, parentStep > 0 {
                throw DeriveError.sourceIsTrainedByLineage(
                    source: sourceName,
                    evidence: "its lineage parent \(parent.modelID) stated trainer step \(parentStep)")
            }
        }
        for derivation in derivationHistory {
            for operation in derivation.operations {
                guard let recorded = operation.arguments[graftSourceTrainingStepArgument] else { continue }
                guard let step = Int(recorded), step >= 0 else {
                    throw DeriveError.malformedSourceTrainingStep(
                        source: sourceName, key: graftSourceTrainingStepArgument, value: recorded)
                }
                if step > 0 {
                    throw DeriveError.sourceIsTrainedByLineage(
                        source: sourceName,
                        evidence: "its derivation history grafts from \(derivation.parentModelID) at source_training_step \(step)")
                }
            }
        }
    }

    /// Refuse unless `target` has exactly `source`'s tensor names and shapes.
    static func requireSameTensorLayout(source: NetworkArchitecture, target: NetworkArchitecture) throws {
        let sourcePlan = source.weightTensorPlan()
        let targetPlan = target.weightTensorPlan()
        guard sourcePlan.count == targetPlan.count else {
            throw DeriveError.shapeChangingRequest(
                detail: "tensor count \(sourcePlan.count) -> \(targetPlan.count)")
        }
        for (before, after) in zip(sourcePlan, targetPlan) where before.name != after.name || before.shape != after.shape {
            throw DeriveError.shapeChangingRequest(
                detail: "'\(before.name)' \(before.shape) -> '\(after.name)' \(after.shape)")
        }
    }

    private static func requireUnchangedOutside(
        _ ranges: [Range<Int>], before: [Float], after: [Float], name: String
    ) throws {
        var allowed = [Bool](repeating: false, count: before.count)
        for range in ranges {
            for index in range where index < allowed.count { allowed[index] = true }
        }
        for index in 0..<before.count where !allowed[index] && before[index].bitPattern != after[index].bitPattern {
            throw DeriveError.rewriteOutsideDeclaredRanges(name: name, elementIndex: index)
        }
    }
}

// MARK: - Operation: set-se-beta-init

/// Sets `se_beta_init` on `scale_and_bias` block groups and re-initializes
/// the β half of each affected block's SE FC2 to match: `zero` writes exact
/// zeros to the β weight rows and β bias; `glorot` re-draws the β weight rows
/// and zeroes the β bias (the builder's bias init). The γ half, and every
/// other tensor, is untouched.
///
/// The `glorot` draw is the graph builder's own: the tensor's
/// `init/<name>` stream under `initSeed` (`WeightInitScheme`), so a derive
/// with a recorded seed is reproducible, and its β rows are exactly the β
/// rows a fresh mint with that init seed would have. The seed and scheme are
/// written into the derivation record.
struct SetSEBetaInitDeriveOperation: DeriveOperation {
    let value: SEBetaInit
    /// 0-based block-group indices, or nil for every `scale_and_bias` group.
    let groupIndices: [Int]?
    /// The init seed of a `glorot` re-draw.
    let initSeed: UInt64

    /// The operation with an explicit init seed (`--init-seed`).
    init(value: SEBetaInit, groupIndices: [Int]?, initSeed: UInt64) {
        self.value = value
        self.groupIndices = groupIndices
        self.initSeed = initSeed
    }

    /// The operation with an init seed drawn for it — the seed is still
    /// recorded, so the derive can be reproduced with `--init-seed`.
    init(value: SEBetaInit, groupIndices: [Int]?) {
        self.init(value: value, groupIndices: groupIndices, initSeed: WeightInitialization.drawnInitSeed())
    }

    /// This operation re-drawing under `seed` instead.
    func withInitSeed(_ seed: UInt64) -> SetSEBetaInitDeriveOperation {
        SetSEBetaInitDeriveOperation(value: value, groupIndices: groupIndices, initSeed: seed)
    }

    /// Whether this operation draws weights (so an init seed applies to it).
    var drawsWeights: Bool { value == .glorot }

    static let kind = DeriveOperationKind(
        name: "set-se-beta-init",
        flag: "--set-se-beta-init",
        valueSyntax: SEBetaInit.allCases.map(\.rawValue).joined(separator: "|"),
        summary: "Set se_beta_init on scale_and_bias block groups (all of them, or those named by --group) "
            + "and re-initialize the β half of each affected block's SE FC2 to match: zero = exact zeros, "
            + "glorot = a fresh Glorot-normal draw (β bias zero) from --init-seed (or a drawn seed, recorded). "
            + "The γ half and every other tensor are copied bit-exact.",
        changedArchitectureFields: ["block_groups[].se_beta_init"],
        rewrittenTensorsDescription: "blocks.<i>.se_scalebias.fc2.weight rows C..2C-1 and "
            + "blocks.<i>.se_scalebias.fc2.bias C..2C-1, for every block i of an affected group",
        acceptsGroupSelection: true,
        make: { value, groupIndices in
            guard let parsed = SEBetaInit(rawValue: value) else {
                throw ModelDerivation.DeriveError.operationNotApplicable(
                    operation: "set-se-beta-init",
                    detail: "value '\(value)' is not one of \(SEBetaInit.allCases.map(\.rawValue).joined(separator: ", "))")
            }
            return SetSEBetaInitDeriveOperation(value: parsed, groupIndices: groupIndices)
        })

    var kindName: String { Self.kind.name }

    var recordedArguments: [String: String] {
        let groups: String
        if let groupIndices {
            groups = groupIndices.map(String.init).joined(separator: ",")
        } else {
            groups = "all scale_and_bias"
        }
        var arguments = ["value": value.rawValue, "groups": groups]
        if drawsWeights {
            arguments["init_seed"] = String(initSeed)
            arguments["init_scheme"] = WeightInitScheme.current
        }
        return arguments
    }

    /// The group indices this operation targets in `architecture`.
    private func selectedGroups(in architecture: NetworkArchitecture) throws -> [Int] {
        if let groupIndices {
            for index in groupIndices where !architecture.blockGroups.indices.contains(index) {
                throw ModelDerivation.DeriveError.operationNotApplicable(
                    operation: kindName,
                    detail: "--group \(index) is out of range (the model has \(architecture.blockGroups.count) block groups, 0-based)")
            }
            for index in groupIndices where architecture.blockGroups[index].seStyle != .scaleAndBias {
                throw ModelDerivation.DeriveError.operationNotApplicable(
                    operation: kindName,
                    detail: "block group \(index) has se_style '\(architecture.blockGroups[index].seStyle.rawValue)'; "
                        + "se_beta_init applies only to '\(SEStyle.scaleAndBias.rawValue)'")
            }
            return groupIndices
        }
        let all = architecture.blockGroups.indices.filter { architecture.blockGroups[$0].seStyle == .scaleAndBias }
        guard !all.isEmpty else {
            throw ModelDerivation.DeriveError.operationNotApplicable(
                operation: kindName, detail: "the model has no '\(SEStyle.scaleAndBias.rawValue)' block group")
        }
        return all
    }

    func apply(to architecture: NetworkArchitecture) throws -> NetworkArchitecture {
        let groups = try selectedGroups(in: architecture)
        guard groups.contains(where: { architecture.blockGroups[$0].seBetaInit != value }) else {
            throw ModelDerivation.DeriveError.operationNotApplicable(
                operation: kindName,
                detail: "every selected block group already has se_beta_init '\(value.rawValue)'; nothing to derive")
        }
        var edited = architecture
        for index in groups { edited.blockGroups[index].seBetaInit = value }
        return edited
    }

    func tensorRewrites(source: NetworkArchitecture, target: NetworkArchitecture) throws -> [DeriveTensorRewrite] {
        let plan = target.weightTensorPlan()
        var planIndexByName: [String: Int] = [:]
        for (index, spec) in plan.enumerated() { planIndexByName[spec.name] = index }

        var rewrites: [DeriveTensorRewrite] = []
        var firstBlock = 0
        for (groupIndex, group) in target.blockGroups.enumerated() {
            defer { firstBlock += group.count }
            guard source.blockGroups[groupIndex].seBetaInit != group.seBetaInit else { continue }
            let channels = group.channels
            let reduced = channels / group.seReductionRatio
            let weightRange = SEScaleAndBiasBetaHalf.torchWeightRange(reducedChannels: reduced, channels: channels)
            let biasRange = SEScaleAndBiasBetaHalf.biasRange(channels: channels)
            for block in firstBlock..<(firstBlock + group.count) {
                let weightName = "blocks.\(block).se_scalebias.fc2.weight"
                let biasName = "blocks.\(block).se_scalebias.fc2.bias"
                guard planIndexByName[weightName] != nil else { throw ModelDerivation.DeriveError.missingTensor(name: weightName) }
                guard planIndexByName[biasName] != nil else { throw ModelDerivation.DeriveError.missingTensor(name: biasName) }
                switch group.seBetaInit {
                case .zero:
                    rewrites.append(DeriveTensorRewrite(
                        tensorName: weightName, rewrittenElementRanges: [weightRange],
                        summary: "β rows \(channels)..<\(2 * channels) zeroed",
                        rewrite: { data in for index in weightRange { data[index] = 0 } }))
                case .glorot:
                    // The builder's own draw for this tensor (native [r, 2C]),
                    // whose β columns are copied into the on-disk [2C, r] β rows.
                    guard let weightIndex = planIndexByName[weightName] else {
                        throw ModelDerivation.DeriveError.missingTensor(name: weightName)
                    }
                    let fresh = try WeightInitScheme.nativeValues(
                        initSeed: initSeed, spec: plan[weightIndex], distribution: .glorotNormal)
                    rewrites.append(DeriveTensorRewrite(
                        tensorName: weightName, rewrittenElementRanges: [weightRange],
                        summary: "β rows \(channels)..<\(2 * channels) re-drawn Glorot-normal",
                        rewrite: { data in
                            for row in 0..<reduced {
                                for betaColumn in 0..<channels {
                                    data[(channels + betaColumn) * reduced + row] = fresh[row * 2 * channels + channels + betaColumn]
                                }
                            }
                        }))
                }
                rewrites.append(DeriveTensorRewrite(
                    tensorName: biasName, rewrittenElementRanges: [biasRange],
                    summary: "β bias \(channels)..<\(2 * channels) zeroed",
                    rewrite: { data in for index in biasRange { data[index] = 0 } }))
            }
        }
        return rewrites
    }
}

// MARK: - Activation values

extension ModelDerivation {

    /// Parses an activation operation's value: one of
    /// `ActivationFunction.functions`. `does_not_apply` is refused with its
    /// own message — it is a marker for a site the topology lacks, never a
    /// function an operation can set — and any other unknown text is refused
    /// naming the functions. The one parser every activation operation uses.
    static func parseActivationFunctionValue(_ value: String, operation: String, refusalReason: String) throws -> ActivationFunction {
        if value == ActivationFunction.doesNotApply.rawValue {
            throw doesNotApplyRefusal(operation: operation, refusalReason: refusalReason)
        }
        guard let parsed = ActivationFunction(rawValue: value) else {
            throw DeriveError.operationNotApplicable(
                operation: operation,
                detail: "value '\(value)' is not one of \(ActivationFunction.functionList)")
        }
        return parsed
    }

    /// The refusal of `does_not_apply` as an operation's value, at parse and
    /// again at apply (an operation constructed directly never reaches the
    /// parser).
    static func doesNotApplyRefusal(operation: String, refusalReason: String) -> DeriveError {
        DeriveError.operationNotApplicable(
            operation: operation,
            detail: "'\(ActivationFunction.doesNotApply.rawValue)' is not an activation function; it marks a site "
                + "the topology lacks, and \(refusalReason)")
    }

    /// `ActivationFunction.functions` as an operation's value syntax.
    static var activationFunctionValueSyntax: String {
        ActivationFunction.functions.map(\.rawValue).joined(separator: "|")
    }
}

// MARK: - Operation: set-activation

/// Sets the main hidden activation everywhere: every architecture-level
/// activation site the model has (stem, tower end, feature-skip fusion,
/// policy head, value conv, value FC1 hidden — each only where the topology
/// has it; an absent site stays `does_not_apply`) and every block group's
/// `activation_function` (block main path, `activation_gated` merge). The
/// rule is `NetworkArchitecture.setMainActivationEverywhere`, which the Build
/// screen's "Use for every activation" applies too. No activation has
/// parameters, so no tensor is rewritten: the derived file holds the
/// source's weights bit-exact and differs only in the activation, which is
/// what an activation A/B from one fresh net needs.
///
/// The SE FC1 activation is a separate field (`se_activation`) with its own
/// operation, `--set-se-activation`, and this one leaves it alone on every
/// group that has an SE block — so "ReLU blocks, leaky FC1" and "leaky
/// everywhere" are both one derive away (the latter = both flags; the
/// catalog order applies this operation first). An SE-less group has no FC1,
/// so its field is `does_not_apply` and stays so. That rule lives in
/// `BlockGroup.setActivationFunction`, which the Build-New-Model screen
/// applies too, so the same activation edit gives the same architecture
/// whichever way it is made.
struct SetActivationDeriveOperation: DeriveOperation {
    let value: ActivationFunction

    /// Why this operation refuses `does_not_apply`: the clause `doesNotApplyRefusal` appends.
    private static let doesNotApplyRefusalReason = "--set-activation sets only sites that exist"

    static let kind = DeriveOperationKind(
        name: "set-activation",
        flag: "--set-activation",
        valueSyntax: ModelDerivation.activationFunctionValueSyntax,
        summary: "Set the main hidden activation everywhere: every architecture-level site the model has (stem, "
            + "tower end, feature-skip fusion, policy head, value conv, value FC1 hidden) and every block group's "
            + "activation_function (block main path, activation_gated merge). A site the topology lacks stays "
            + "does_not_apply. The SE FC1 activation is not changed (use --set-se-activation; an SE-less "
            + "group's se_activation is does_not_apply). Activations have no parameters, so every tensor is "
            + "copied bit-exact.",
        changedArchitectureFields: ArchitectureActivationSite.allCases.map(\.jsonKey) + [
            "block_groups[].activation_function",
        ],
        rewrittenTensorsDescription: "none",
        acceptsGroupSelection: false,
        make: { value, _ in
            SetActivationDeriveOperation(value: try ModelDerivation.parseActivationFunctionValue(
                value, operation: "set-activation", refusalReason: SetActivationDeriveOperation.doesNotApplyRefusalReason))
        })

    var kindName: String { Self.kind.name }

    var recordedArguments: [String: String] { ["value": value.rawValue] }

    func apply(to architecture: NetworkArchitecture) throws -> NetworkArchitecture {
        guard value != .doesNotApply else {
            throw ModelDerivation.doesNotApplyRefusal(operation: kindName, refusalReason: Self.doesNotApplyRefusalReason)
        }
        let alreadySet = ArchitectureActivationSite.allCases
            .filter { architecture.hasActivationSite($0) }
            .allSatisfy { architecture.activation(at: $0) == value }
            && architecture.blockGroups.allSatisfy { $0.activationFunction == value }
        guard !alreadySet else {
            throw ModelDerivation.DeriveError.operationNotApplicable(
                operation: kindName,
                detail: "every existing architecture-level activation site and every group's activation_function "
                    + "are already '\(value.rawValue)'; nothing to derive")
        }
        var edited = architecture
        try edited.setMainActivationEverywhere(value)
        return edited
    }

    func tensorRewrites(source: NetworkArchitecture, target: NetworkArchitecture) throws -> [DeriveTensorRewrite] {
        []
    }
}

// MARK: - Operation: set-se-activation

/// Sets `se_activation` — the activation after the SE excitation FC1 — on
/// block groups that have an SE block (GitHub issue #2). The motivating A/B
/// is ReLU vs leaky ReLU at FC1 from one fresh net, comparing dead FC1 units
/// and strength, with the rest of the network (including the main-path
/// activation) unchanged. No activation has parameters, so no tensor is
/// rewritten: the derived file holds the source's weights bit-exact.
struct SetSEActivationDeriveOperation: DeriveOperation {
    let value: ActivationFunction
    /// 0-based block-group indices, or nil for every group with an SE block.
    let groupIndices: [Int]?

    /// Why this operation refuses `does_not_apply`: the clause `doesNotApplyRefusal` appends.
    private static let doesNotApplyRefusalReason = "an SE FC1 exists whenever its SE block does"

    static let kind = DeriveOperationKind(
        name: "set-se-activation",
        flag: "--set-se-activation",
        valueSyntax: ModelDerivation.activationFunctionValueSyntax,
        summary: "Set se_activation, the activation after the SE excitation FC1, on block groups that have an "
            + "SE block (all of them, or those named by --group). The main-path activation is not changed. "
            + "Activations have no parameters, so every tensor is copied bit-exact.",
        changedArchitectureFields: ["block_groups[].se_activation"],
        rewrittenTensorsDescription: "none",
        acceptsGroupSelection: true,
        make: { value, groupIndices in
            SetSEActivationDeriveOperation(
                value: try ModelDerivation.parseActivationFunctionValue(
                    value, operation: "set-se-activation", refusalReason: SetSEActivationDeriveOperation.doesNotApplyRefusalReason),
                groupIndices: groupIndices)
        })

    var kindName: String { Self.kind.name }

    var recordedArguments: [String: String] {
        let groups: String
        if let groupIndices {
            groups = groupIndices.map(String.init).joined(separator: ",")
        } else {
            groups = "all with SE"
        }
        return ["value": value.rawValue, "groups": groups]
    }

    /// The group indices this operation targets in `architecture`.
    private func selectedGroups(in architecture: NetworkArchitecture) throws -> [Int] {
        if let groupIndices {
            for index in groupIndices where !architecture.blockGroups.indices.contains(index) {
                throw ModelDerivation.DeriveError.operationNotApplicable(
                    operation: kindName,
                    detail: "--group \(index) is out of range (the model has \(architecture.blockGroups.count) block groups, 0-based)")
            }
            for index in groupIndices where !architecture.blockGroups[index].hasSEFC1 {
                throw ModelDerivation.DeriveError.operationNotApplicable(
                    operation: kindName,
                    detail: "block group \(index) has se_style '\(architecture.blockGroups[index].seStyle.rawValue)'; se_activation applies only "
                        + "to a group with an SE block")
            }
            return groupIndices
        }
        let all = architecture.blockGroups.indices.filter { architecture.blockGroups[$0].hasSEFC1 }
        guard !all.isEmpty else {
            throw ModelDerivation.DeriveError.operationNotApplicable(
                operation: kindName, detail: "the model has no block group with an SE block")
        }
        return all
    }

    func apply(to architecture: NetworkArchitecture) throws -> NetworkArchitecture {
        guard value != .doesNotApply else {
            throw ModelDerivation.doesNotApplyRefusal(operation: kindName, refusalReason: Self.doesNotApplyRefusalReason)
        }
        let groups = try selectedGroups(in: architecture)
        guard groups.contains(where: { architecture.blockGroups[$0].seActivation != value }) else {
            throw ModelDerivation.DeriveError.operationNotApplicable(
                operation: kindName,
                detail: "every selected block group already has se_activation '\(value.rawValue)'; nothing to derive")
        }
        var edited = architecture
        for index in groups { edited.blockGroups[index].seActivation = value }
        return edited
    }

    func tensorRewrites(source: NetworkArchitecture, target: NetworkArchitecture) throws -> [DeriveTensorRewrite] {
        []
    }
}

// MARK: - ReZero operations: shared parsing and group selection

/// What `--set-rezero-alpha-init` and `--set-rezero-alpha-cap` share: parsing
/// the numeric value and choosing the block groups. Both apply only to groups
/// with ReZero — on a group without it neither value is read by anything, so
/// setting one there would be a silent no-op the user almost certainly did
/// not mean.
enum RezeroDeriveSupport {

    /// The flag value as a Float. Range checks are not done here: the derived
    /// architecture goes through `NetworkArchitecture.validate()`, the one
    /// place that decides what a legal init and cap are, so a value it rejects
    /// is refused with the same error the Build screen and the loaders show.
    static func parseValue(_ value: String, operation: String) throws -> Float {
        guard let parsed = Float(value) else {
            throw ModelDerivation.DeriveError.operationNotApplicable(
                operation: operation, detail: "value '\(value)' is not a number")
        }
        return parsed
    }

    /// The group indices an operation targets in `architecture`: the given
    /// ones (each must exist and have ReZero), or every ReZero group.
    static func selectedGroups(_ groupIndices: [Int]?, in architecture: NetworkArchitecture, operation: String) throws -> [Int] {
        if let groupIndices {
            for index in groupIndices where !architecture.blockGroups.indices.contains(index) {
                throw ModelDerivation.DeriveError.operationNotApplicable(
                    operation: operation,
                    detail: "--group \(index) is out of range (the model has \(architecture.blockGroups.count) block groups, 0-based)")
            }
            for index in groupIndices where !architecture.blockGroups[index].useRezero {
                throw ModelDerivation.DeriveError.operationNotApplicable(
                    operation: operation,
                    detail: "block group \(index) has use_rezero false; the ReZero init and cap apply only to a group with ReZero")
            }
            return groupIndices
        }
        let all = architecture.blockGroups.indices.filter { architecture.blockGroups[$0].useRezero }
        guard !all.isEmpty else {
            throw ModelDerivation.DeriveError.operationNotApplicable(
                operation: operation, detail: "the model has no block group with ReZero (use_rezero true)")
        }
        return all
    }

    /// The `groups` argument recorded in the derivation history.
    static func recordedGroups(_ groupIndices: [Int]?) -> String {
        guard let groupIndices else { return "all with ReZero" }
        return groupIndices.map(String.init).joined(separator: ",")
    }
}

// MARK: - Operation: set-rezero-alpha-init

/// Sets `rezero_alpha_init` on block groups with ReZero AND rewrites every
/// `blocks.<i>.rezero_alpha` tensor of those groups to exactly that value.
/// The init is a tensor value: the architecture field only says what a
/// random-weights build starts α at, so changing the field alone would leave
/// a file whose α tensors still hold the old start and describe a net that
/// was never built. The motivating case is the zero init (the ReZero paper's):
/// a fresh net's α tensors set to exactly 0, every other tensor bit-exact, so
/// "zero-init vs 1/√N" is a paired A/B from one net.
///
/// The cap (`rezero_alpha_cap`) is not touched — a legacy group's cap stays
/// at its old init, which keeps the forward well-defined (the init alone may
/// be zero; the cap may not). Pair with `--set-rezero-alpha-cap` to change
/// both.
///
/// Optimizer velocity is not handled here because it cannot reach here:
/// `ModelDerivation.derive` refuses any source carrying optimizer state
/// (`opt.*` tensors or trainer-schedule metadata). That refusal is the right
/// answer for α in particular — velocity accumulated toward the old α would
/// push the reset value back toward it on the first step.
///
/// "Nothing to derive" is judged on the field: a request whose value every
/// selected group already states is refused, even if the file's α tensors
/// hold something else. A trained source never reaches the rewrite at all:
/// `ModelDerivation.derive` refuses any tensor rewrite on a source that
/// shows training (`ModelDerivation.requireUntrainedSource`).
struct SetRezeroAlphaInitDeriveOperation: DeriveOperation {
    let value: Float
    /// 0-based block-group indices, or nil for every group with ReZero.
    let groupIndices: [Int]?

    static let kind = DeriveOperationKind(
        name: "set-rezero-alpha-init",
        flag: "--set-rezero-alpha-init",
        valueSyntax: "<float >= 0>",
        summary: "Set rezero_alpha_init on block groups with ReZero (all of them, or those named by --group) and "
            + "rewrite every block's ReZero alpha tensor in those groups to exactly that value; 0 is the ReZero "
            + "paper's init (every residual branch starts off). The cap (rezero_alpha_cap) is unchanged — pair "
            + "with --set-rezero-alpha-cap to set it. Every other tensor is copied bit-exact.",
        changedArchitectureFields: ["block_groups[].rezero_alpha_init"],
        rewrittenTensorsDescription: "blocks.<i>.rezero_alpha (the whole one-element tensor), for every block i of an affected group",
        acceptsGroupSelection: true,
        make: { value, groupIndices in
            SetRezeroAlphaInitDeriveOperation(
                value: try RezeroDeriveSupport.parseValue(value, operation: "set-rezero-alpha-init"),
                groupIndices: groupIndices)
        })

    var kindName: String { Self.kind.name }

    var recordedArguments: [String: String] {
        ["value": "\(value)", "groups": RezeroDeriveSupport.recordedGroups(groupIndices)]
    }

    func apply(to architecture: NetworkArchitecture) throws -> NetworkArchitecture {
        let groups = try RezeroDeriveSupport.selectedGroups(groupIndices, in: architecture, operation: kindName)
        guard groups.contains(where: { architecture.blockGroups[$0].rezeroAlphaInit.bitPattern != value.bitPattern }) else {
            throw ModelDerivation.DeriveError.operationNotApplicable(
                operation: kindName,
                detail: "every selected block group already has rezero_alpha_init \(value); nothing to derive")
        }
        var edited = architecture
        for index in groups { edited.blockGroups[index].rezeroAlphaInit = value }
        return edited
    }

    /// One rewrite per block of every selected group — including a selected
    /// group whose field already held `value`, because its α tensors are
    /// what the init means and they may differ from the field.
    func tensorRewrites(source: NetworkArchitecture, target: NetworkArchitecture) throws -> [DeriveTensorRewrite] {
        let selected = Set(try RezeroDeriveSupport.selectedGroups(groupIndices, in: source, operation: kindName))
        var planNames: Set<String> = []
        for spec in target.weightTensorPlan() { planNames.insert(spec.name) }

        let newValue = value
        var rewrites: [DeriveTensorRewrite] = []
        var firstBlock = 0
        for (groupIndex, group) in target.blockGroups.enumerated() {
            defer { firstBlock += group.count }
            guard selected.contains(groupIndex) else { continue }
            for block in firstBlock..<(firstBlock + group.count) {
                let name = "blocks.\(block).rezero_alpha"
                guard planNames.contains(name) else { throw ModelDerivation.DeriveError.missingTensor(name: name) }
                rewrites.append(DeriveTensorRewrite(
                    tensorName: name, rewrittenElementRanges: [0..<1],
                    summary: "alpha set to \(newValue)",
                    rewrite: { data in
                        for index in data.indices { data[index] = newValue }
                    }))
            }
        }
        return rewrites
    }
}

// MARK: - Operation: set-rezero-alpha-cap

/// Sets `rezero_alpha_cap` — the asymptote C of the forward soft bound
/// `C·tanh(α/C)` — on block groups with ReZero. The cap is not a parameter:
/// no tensor is rewritten, and the derived file holds the source's weights
/// bit-exact, so a cap A/B from one fresh net differs in exactly the bound.
struct SetRezeroAlphaCapDeriveOperation: DeriveOperation {
    let value: Float
    /// 0-based block-group indices, or nil for every group with ReZero.
    let groupIndices: [Int]?

    static let kind = DeriveOperationKind(
        name: "set-rezero-alpha-cap",
        flag: "--set-rezero-alpha-cap",
        valueSyntax: "<float > 0>",
        summary: "Set rezero_alpha_cap, the asymptote C of the forward ReZero soft bound C*tanh(alpha/C), on "
            + "block groups with ReZero (all of them, or those named by --group). The cap has no parameters, "
            + "so every tensor is copied bit-exact.",
        changedArchitectureFields: ["block_groups[].rezero_alpha_cap"],
        rewrittenTensorsDescription: "none",
        acceptsGroupSelection: true,
        make: { value, groupIndices in
            SetRezeroAlphaCapDeriveOperation(
                value: try RezeroDeriveSupport.parseValue(value, operation: "set-rezero-alpha-cap"),
                groupIndices: groupIndices)
        })

    var kindName: String { Self.kind.name }

    var recordedArguments: [String: String] {
        ["value": "\(value)", "groups": RezeroDeriveSupport.recordedGroups(groupIndices)]
    }

    func apply(to architecture: NetworkArchitecture) throws -> NetworkArchitecture {
        let groups = try RezeroDeriveSupport.selectedGroups(groupIndices, in: architecture, operation: kindName)
        guard groups.contains(where: { architecture.blockGroups[$0].rezeroAlphaCap.bitPattern != value.bitPattern }) else {
            throw ModelDerivation.DeriveError.operationNotApplicable(
                operation: kindName,
                detail: "every selected block group already has rezero_alpha_cap \(value); nothing to derive")
        }
        var edited = architecture
        for index in groups { edited.blockGroups[index].rezeroAlphaCap = value }
        return edited
    }

    func tensorRewrites(source: NetworkArchitecture, target: NetworkArchitecture) throws -> [DeriveTensorRewrite] {
        []
    }
}
