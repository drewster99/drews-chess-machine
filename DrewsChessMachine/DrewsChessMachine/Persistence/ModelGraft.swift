//
//  ModelGraft.swift
//  DrewsChessMachine
//
//  `--derive-model --graft-to`: the one derive that may change the tensor
//  layout. It makes a model of a TARGET architecture from a source model:
//  every target tensor whose name (after `--graft-map` renames) and on-disk
//  shape match a source tensor is copied bit-exact; every other target
//  tensor gets the value a fresh mint of the target would give it under the
//  graft's init seed. The motivating case is warm-starting a larger tower
//  from a trained smaller one — five trained blocks plus a sixth new one —
//  without a retrain, while the result still says exactly which tensors are
//  learned and which are new.
//
//  Why the target's fresh values come from a network build rather than a
//  CPU formula: the graph builder is the only definition of every tensor's
//  initial value (the drawn conv / FC weights, AND the constants — BN γ/β,
//  running statistics, biases, ReZero α, the W/D/L prior). `GraftFreshTarget`
//  builds the target with `.seeded(initSeed)` and reads those values back,
//  so an initialized tensor is bit-identical to the same tensor of a fresh
//  `--new-model` mint with that seed (minus `--new-model`'s BN warm-up, which
//  a graft does not run: a new BN layer keeps the builder's identity running
//  statistics, mean 0 / variance 1, and the record says so). The build reads
//  variables only — no forward pass, no training.
//
//  Trained sources. The in-place derive operations refuse a trained source
//  because their output keeps the source's `training_step` and lineage while
//  some of its weights were reset. A graft's output keeps NO `training_step`
//  and lists every tensor it initialized, so it claims nothing its weights do
//  not have; a trained source is therefore allowed — it is the point. The
//  source's step is recorded in the derivation record
//  (`source_training_step`), and the lineage continues the source's totals
//  (the copied weights carry that history) as a new derived run. That record
//  and lineage are also what keep a graft of a trained source trained in a
//  later derive's eyes: the in-place operations refuse a source whose
//  lineage or derivation history shows training, not only one whose
//  `training_step` does (`ModelDerivation.requireUntrainedSource`).
//
//  Guardrails: the source must be a plain model file that the full loader
//  accepts; the target must validate; a `--graft-map` entry must name a real
//  source tensor and a real target tensor of the same on-disk shape; a
//  target tensor that a source tensor would fill by name but with a
//  different shape is refused rather than silently re-initialized (drop it
//  explicitly with `name=`); the output must load back as the target, and
//  every copied tensor is verified bit-identical in the written file.
//

import CryptoKit
import Foundation

// MARK: - Fresh target

/// The target architecture's tensors as a fresh mint under `initSeed` would
/// hold them, read from a seeded network build. Only `build` makes one, so
/// the values, the seed and the architecture cannot disagree.
struct GraftFreshTarget: Sendable {
    let architecture: NetworkArchitecture
    let initSeed: UInt64
    /// On-disk (PyTorch state_dict layout) tensors, in plan order.
    let tensors: [SafetensorsTensor]
    /// The distribution each drawn tensor came from, by plan name.
    let randomTensorRoles: [String: RandomTensorRole]

    private init(architecture: NetworkArchitecture, initSeed: UInt64, tensors: [SafetensorsTensor],
                 randomTensorRoles: [String: RandomTensorRole]) {
        self.architecture = architecture
        self.initSeed = initSeed
        self.tensors = tensors
        self.randomTensorRoles = randomTensorRoles
    }

    /// Build `architecture` with `.seeded(initSeed)` (inference BN mode, no
    /// BN warm-up) and read its tensors back. Blocks the calling thread for
    /// the build and one variable read; call it from a CLI thread or a
    /// dispatch queue, never from the Swift concurrency pool.
    static func build(architecture: NetworkArchitecture, initSeed: UInt64) throws -> GraftFreshTarget {
        let network = try ChessNetwork(arch: architecture, bnMode: .inference,
                                       initialization: .seeded(initSeed: initSeed))
        let weights = try network.exportWeightsBlocking()
        let plan = architecture.weightTensorPlan()
        guard weights.count == plan.count else {
            throw ModelDerivation.GraftError.freshTargetIncomplete(exported: weights.count, plan: plan.count)
        }
        var tensors: [SafetensorsTensor] = []
        tensors.reserveCapacity(plan.count)
        for (spec, native) in zip(plan, weights) {
            let (shape, data) = SafetensorsModelIO.toTorchLayout(kind: spec.kind, nativeShape: spec.shape, data: native)
            tensors.append(SafetensorsTensor(name: spec.name, shape: shape, data: data))
        }
        return GraftFreshTarget(architecture: architecture, initSeed: initSeed, tensors: tensors,
                                randomTensorRoles: network.randomTensorRoles)
    }
}

// MARK: - Graft map

/// `--graft-map` parsed: how source tensor names move into the target.
///
/// Syntax: comma-separated `old=new` pairs.
/// - `old=new` renames one source tensor (exact names).
/// - `old.=new.` — both sides ending in `.` — renames every source tensor
///   whose name starts with `old.` (e.g. `blocks.2.=blocks.3.` moves a
///   whole block when one is inserted before it).
/// - `old=` (empty right side) drops that source tensor, so a target tensor
///   of the same name but a different shape is initialized instead of
///   refused.
/// A source tensor matched by more than one entry is refused, never
/// resolved by precedence.
struct GraftMap: Sendable, Equatable {
    struct Entry: Sendable, Equatable {
        let old: String
        /// nil = drop.
        let new: String?
        var isPrefix: Bool { old.hasSuffix(".") }
    }
    let entries: [Entry]

    static let empty = GraftMap(entries: [])

    /// The text as recorded (`old=new,…`, in the given order).
    var recordedText: String {
        entries.map { "\($0.old)=\($0.new ?? "")" }.joined(separator: ",")
    }

    static func parse(_ text: String) throws -> GraftMap {
        var entries: [Entry] = []
        for raw in text.split(separator: ",", omittingEmptySubsequences: false) {
            let pair = raw.trimmingCharacters(in: .whitespaces)
            let parts = pair.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count == 2, !parts[0].isEmpty else {
                throw ModelDerivation.GraftError.badGraftMap(detail: "'\(pair)' is not old=new (or old= to drop)")
            }
            let old = String(parts[0])
            let new = parts[1].isEmpty ? nil : String(parts[1])
            if let new, old.hasSuffix(".") != new.hasSuffix(".") {
                throw ModelDerivation.GraftError.badGraftMap(
                    detail: "'\(pair)': a prefix rename needs both sides to end in '.'")
            }
            guard !entries.contains(where: { $0.old == old }) else {
                throw ModelDerivation.GraftError.badGraftMap(detail: "'\(old)' appears twice")
            }
            entries.append(Entry(old: old, new: new))
        }
        return GraftMap(entries: entries)
    }

    /// Where `sourceName` goes: `.unchanged`, `.renamed(to:)` or `.dropped`.
    enum Destination: Equatable { case unchanged, renamed(String), dropped }

    func destination(of sourceName: String) throws -> Destination {
        let matches = entries.filter { $0.isPrefix ? sourceName.hasPrefix($0.old) : sourceName == $0.old }
        guard matches.count <= 1 else {
            throw ModelDerivation.GraftError.badGraftMap(
                detail: "source tensor '\(sourceName)' is matched by \(matches.map(\.old).joined(separator: " and "))")
        }
        guard let entry = matches.first else { return .unchanged }
        guard let new = entry.new else { return .dropped }
        if entry.isPrefix {
            return .renamed(new + sourceName.dropFirst(entry.old.count))
        }
        return .renamed(new)
    }
}

// MARK: - Engine

extension ModelDerivation {

    /// The derivation-record operation name of a graft.
    static let graftOperationName = "graft"
    /// The graft record's argument stating the source file's
    /// `training_step` — the output carries none of its own, so this is
    /// where a later derive learns the grafted weights were trained.
    static let graftSourceTrainingStepArgument = "source_training_step"
    /// `per_tensor_init` value of an initialized tensor the builder sets to
    /// a fixed value (biases, BN, ReZero α, priors) rather than drawing.
    static let builderConstantInit = "builder_constant"

    enum GraftError: Error, CustomStringConvertible, Equatable {
        case freshTargetIncomplete(exported: Int, plan: Int)
        case freshTargetMismatch(detail: String)
        case badGraftMap(detail: String)
        case mapNamesUnknownSourceTensor(entry: String)
        case mapNamesUnknownTargetTensor(source: String, target: String)
        case mapShapeMismatch(source: String, sourceShape: [Int], target: String, targetShape: [Int])
        case sameNameShapeMismatch(name: String, sourceShape: [Int], targetShape: [Int])
        case twoSourcesForOneTarget(target: String, sources: [String])
        case copiedTensorChangedInOutput(name: String)

        var description: String {
            switch self {
            case let .freshTargetIncomplete(exported, plan):
                return "the target build exported \(exported) tensors but its plan names \(plan)"
            case let .freshTargetMismatch(detail):
                return "internal error: the fresh target does not match the graft target (\(detail))"
            case let .badGraftMap(detail):
                return "--graft-map: \(detail)"
            case let .mapNamesUnknownSourceTensor(entry):
                return "--graft-map entry '\(entry)' matches no tensor of the source model"
            case let .mapNamesUnknownTargetTensor(source, target):
                return "--graft-map moves '\(source)' to '\(target)', which the target architecture does not have"
            case let .mapShapeMismatch(source, sourceShape, target, targetShape):
                return "--graft-map moves '\(source)' \(sourceShape) to '\(target)' \(targetShape): shapes differ"
            case let .sameNameShapeMismatch(name, sourceShape, targetShape):
                return "'\(name)' is \(sourceShape) in the source but \(targetShape) in the target; refusing to "
                    + "re-initialize a source tensor silently — add '\(name)=' to --graft-map to drop it and "
                    + "initialize the target's"
            case let .twoSourcesForOneTarget(target, sources):
                return "target tensor '\(target)' would be filled by \(sources.joined(separator: " and "))"
            case let .copiedTensorChangedInOutput(name):
                return "the written file's '\(name)' differs from the source tensor it was copied from"
            }
        }
    }

    /// What `graft` produced.
    struct GraftResult: Sendable {
        let data: Data
        let sourceArchitecture: NetworkArchitecture
        let targetArchitecture: NetworkArchitecture
        let record: DerivationRecord
        let lineage: LineageRecord
        let copied: [String]
        let initialized: [String]
        let dropped: [String]
        let sourceArchitectureFormat: ArchitectureFormat.DecodeFormat
    }

    /// Graft the model in `sourceData` onto `fresh.architecture`. Pure: no
    /// file, log or GPU work (`fresh` was built beforehand).
    static func graft(
        sourceData: Data,
        sourceName: String,
        fresh: GraftFreshTarget,
        targetLabel: String,
        map: GraftMap,
        initSeedOrigin: String,
        newModelID: String,
        createdAtUnix: Int64,
        build: String,
        invocationArguments: [String]
    ) throws -> GraftResult {
        let (sourceTensors, sourceMetadata) = try SafetensorsFile.decode(sourceData)
        let parentModelID = try requirePlainModelSource(tensors: sourceTensors, metadata: sourceMetadata,
                                                        sourceName: sourceName)
        // The full loader must accept the source (plan, dims, value-head marker).
        let sourceDecoded = try SafetensorsModelIO.decode(sourceData, valueHead: .asStored, source: sourceName)

        let target = fresh.architecture
        do {
            try target.validate()
        } catch {
            throw DeriveError.invalidTargetArchitecture(detail: String(describing: error))
        }
        let plan = target.weightTensorPlan()
        guard fresh.tensors.count == plan.count else {
            throw GraftError.freshTargetMismatch(detail: "\(fresh.tensors.count) tensors for a \(plan.count)-tensor plan")
        }
        for (spec, tensor) in zip(plan, fresh.tensors)
        where tensor.name != spec.name || tensor.shape != SafetensorsModelIO.torchShape(for: spec) {
            throw GraftError.freshTargetMismatch(detail: "'\(tensor.name)' \(tensor.shape) vs plan '\(spec.name)'")
        }
        var targetIndexByName: [String: Int] = [:]
        for (index, tensor) in fresh.tensors.enumerated() { targetIndexByName[tensor.name] = index }

        // Every map entry must match something in the source.
        for entry in map.entries {
            let matches = sourceTensors.contains { entry.isPrefix ? $0.name.hasPrefix(entry.old) : $0.name == entry.old }
            guard matches else { throw GraftError.mapNamesUnknownSourceTensor(entry: entry.old) }
        }

        // Route each source tensor to its target slot (or drop it).
        var sourceForTarget: [String: [SafetensorsTensor]] = [:]
        var renamedFrom: [String: String] = [:]
        var dropped: [String] = []
        for tensor in sourceTensors {
            switch try map.destination(of: tensor.name) {
            case .dropped:
                dropped.append(tensor.name)
            case .unchanged:
                if let index = targetIndexByName[tensor.name] {
                    let targetShape = fresh.tensors[index].shape
                    guard targetShape == tensor.shape else {
                        throw GraftError.sameNameShapeMismatch(
                            name: tensor.name, sourceShape: tensor.shape, targetShape: targetShape)
                    }
                    sourceForTarget[tensor.name, default: []].append(tensor)
                } else {
                    dropped.append(tensor.name)
                }
            case .renamed(let targetName):
                guard let index = targetIndexByName[targetName] else {
                    throw GraftError.mapNamesUnknownTargetTensor(source: tensor.name, target: targetName)
                }
                let targetShape = fresh.tensors[index].shape
                guard targetShape == tensor.shape else {
                    throw GraftError.mapShapeMismatch(
                        source: tensor.name, sourceShape: tensor.shape, target: targetName, targetShape: targetShape)
                }
                sourceForTarget[targetName, default: []].append(tensor)
                renamedFrom[targetName] = tensor.name
            }
        }
        for (targetName, sources) in sourceForTarget where sources.count > 1 {
            throw GraftError.twoSourcesForOneTarget(target: targetName, sources: sources.map(\.name).sorted())
        }

        // Assemble the output in plan order.
        var outputTensors: [SafetensorsTensor] = []
        outputTensors.reserveCapacity(plan.count)
        var copied: [String] = []
        var copiedRecord: [String] = []
        var initialized: [String] = []
        var perTensorInit: [String: String] = [:]
        for (spec, freshTensor) in zip(plan, fresh.tensors) {
            if let source = sourceForTarget[freshTensor.name]?.first {
                outputTensors.append(SafetensorsTensor(name: freshTensor.name, shape: freshTensor.shape, data: source.data))
                copied.append(freshTensor.name)
                if let from = renamedFrom[freshTensor.name] {
                    copiedRecord.append("\(from) -> \(freshTensor.name)")
                } else {
                    copiedRecord.append(freshTensor.name)
                }
            } else {
                outputTensors.append(freshTensor)
                initialized.append(freshTensor.name)
                switch spec.kind {
                case .conv, .linear:
                    // Every conv / FC weight is drawn, and the build records
                    // how; one without a recorded role is a build defect.
                    guard let role = fresh.randomTensorRoles[freshTensor.name] else {
                        throw GraftError.freshTargetMismatch(detail: "no recorded draw for '\(freshTensor.name)'")
                    }
                    perTensorInit[freshTensor.name] = role.rawValue
                case .bias, .bnAffine, .bnRunningStat, .scalar:
                    perTensorInit[freshTensor.name] = builderConstantInit
                }
            }
        }

        // The record. The source's training step is stated here because the
        // output does not carry it (see the file header).
        var arguments: [String: String] = [
            "target": targetLabel,
            "init_seed_origin": initSeedOrigin,
            "bn_running_stats": "copied BN layers keep the source's; new BN layers start at the builder's "
                + "identity statistics (mean 0, variance 1), not recalibrated",
        ]
        if !map.entries.isEmpty { arguments["graft_map"] = map.recordedText }
        if let step = sourceMetadata[SafetensorsModelIO.Key.trainingStep] {
            arguments[graftSourceTrainingStepArgument] = step
        }
        var operation = OperationRecord(
            operation: graftOperationName, arguments: arguments,
            changedArchitectureFields: ["*"], rewrittenTensors: initialized)
        operation.copiedTensors = copiedRecord
        operation.droppedTensors = dropped
        operation.initSeed = String(fresh.initSeed)
        operation.initRuleVersion = WeightInitScheme.current
        operation.perTensorInit = perTensorInit
        let record = DerivationRecord(
            modelID: newModelID,
            parentModelID: parentModelID,
            sourceFile: sourceName,
            sourceSHA256: SHA256.hash(data: sourceData).map { String(format: "%02x", $0) }.joined(),
            sourceFormatVersion: sourceDecoded.architectureFormat.formatVersion,
            createdAtUnix: createdAtUnix,
            build: build,
            operations: [operation])

        // Lineage: a new derived run continuing the source's totals.
        let sourceLineage = try SafetensorsModelIO.lineage(
            fromMetadata: sourceMetadata, formatVersion: sourceDecoded.architectureFormat.formatVersion)
        let sourceParent = LineageTracker.ParentFile(
            modelID: parentModelID,
            contentSHA256: sourceMetadata[SafetensorsFile.contentHashKey],
            trainerCompletedSteps: try SafetensorsModelIO.trainerClock(fromMetadata: sourceMetadata, source: sourceName),
            lineage: sourceLineage,
            derivationHistory: try LineageTracker.ParentFile.derivationHistory(lineage: sourceLineage, metadata: sourceMetadata))
        let lineage = LineageTracker.untrainedCopyRecord(
            source: sourceParent, derivation: record, pathKind: .derive, argv: invocationArguments,
            at: Date(timeIntervalSince1970: TimeInterval(createdAtUnix)))

        // Metadata is built, not copied: a graft's file states only what is
        // true of the grafted weights. The value-head marker travels with the
        // source's own state (as for every derive), so an unmarked source's
        // copied head is recentered on load exactly as the source's would be.
        var metadata: [String: String] = [
            SafetensorsModelIO.Key.formatVersion: SafetensorsModelIO.formatVersion,
            SafetensorsModelIO.Key.modelID: newModelID,
            SafetensorsModelIO.Key.createdAt: String(createdAtUnix),
            SafetensorsModelIO.Key.creator: creator,
            SafetensorsModelIO.Key.parentModelID: parentModelID,
            SafetensorsModelIO.Key.notes: graftNotes(record: record, targetLabel: targetLabel, copied: copied.count,
                                                     initialized: initialized.count, dropped: dropped.count),
            SafetensorsModelIO.Key.architecture: String(decoding: try JSONEncoder().encode(target), as: UTF8.self),
        ]
        if let marker = sourceMetadata[ValueHeadRecentering.metadataKey] {
            metadata[ValueHeadRecentering.metadataKey] = marker
        }
        for (key, value) in try lineage.metadataEntries() { metadata[key] = value }

        let data = try SafetensorsFile.encode(tensors: outputTensors, metadata: metadata)

        // The output must load back as the target, and every copied tensor
        // must be bit-identical in the written file.
        do {
            let reloaded = try SafetensorsModelIO.decode(data, valueHead: .asStored, source: "grafted output")
            guard reloaded.architecture == target else {
                throw DeriveError.outputFailedVerification(detail: "embedded architecture differs from the target")
            }
        } catch let error as DeriveError {
            throw error
        } catch {
            throw DeriveError.outputFailedVerification(detail: String(describing: error))
        }
        let (writtenTensors, _) = try SafetensorsFile.decode(data)
        var writtenByName: [String: SafetensorsTensor] = [:]
        for tensor in writtenTensors { writtenByName[tensor.name] = tensor }
        for name in copied {
            guard let written = writtenByName[name], let source = sourceForTarget[name]?.first,
                  written.shape == source.shape, written.data.count == source.data.count,
                  zip(written.data, source.data).allSatisfy({ $0.bitPattern == $1.bitPattern }) else {
                throw GraftError.copiedTensorChangedInOutput(name: name)
            }
        }

        return GraftResult(
            data: data,
            sourceArchitecture: sourceDecoded.architecture,
            targetArchitecture: target,
            record: record,
            lineage: lineage,
            copied: copied,
            initialized: initialized,
            dropped: dropped,
            sourceArchitectureFormat: sourceDecoded.architectureFormat)
    }

    /// The `notes` line of a grafted file.
    static func graftNotes(record: DerivationRecord, targetLabel: String, copied: Int, initialized: Int,
                           dropped: Int) -> String {
        "grafted from \(record.parentModelID) (\(record.sourceFile), sha256 \(record.sourceSHA256)) onto "
            + "\(targetLabel): \(copied) tensors copied, \(initialized) initialized, \(dropped) dropped"
    }
}
