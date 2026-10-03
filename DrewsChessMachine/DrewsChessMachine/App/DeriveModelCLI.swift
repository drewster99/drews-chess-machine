//
//  DeriveModelCLI.swift
//
//  Headless `--derive-model` pre-flight: read one model file, apply the
//  requested derive operations (`ModelDerivation.operationKinds`), write the
//  derived model to a new `.safetensors`, print its path, exit. The
//  same-layout operations are a pure file transform (no GPU, no network
//  build); a graft (`--graft-to`) builds the target network once to read its
//  freshly initialized tensors (no forward pass, no training), like
//  `--new-model`. Both are safe to run next to a training job.
//
//      DrewsChessMachine --derive-model --from <model.safetensors>
//          <operation flag> <value> [--group <index>]... --out <file.safetensors>
//      DrewsChessMachine --derive-model --from <model.safetensors>
//          --graft-to <preset name | arch.json> [--graft-map <old=new,...>] [--init-seed <u64>]
//          --out <file.safetensors>
//      DrewsChessMachine --derive-model --help
//
//  The operation flags are not hard-coded here: every flag, its value syntax
//  and its `--help` text come from `ModelDerivation.operationKinds`, so adding
//  an operation there is all it takes to expose it. See
//  `documentation/deriving-models.md` for examples.
//

import Foundation

enum DeriveModelCLI {

    static let flag = "--derive-model"
    static let fromFlag = "--from"
    static let outFlag = "--out"
    static let groupFlag = "--group"
    static let initSeedFlag = "--init-seed"
    static let graftToFlag = "--graft-to"
    static let graftMapFlag = "--graft-map"
    static let helpFlag = "--help"

    /// `--derive-model --help` text, generated from the operation catalog.
    static var helpText: String {
        var lines: [String] = [
            "usage: DrewsChessMachine \(flag) \(fromFlag) <model.safetensors> <operation> <value> "
                + "[\(groupFlag) <index>]... \(outFlag) <new.safetensors>",
            "",
            "Writes a new model whose weights are copied bit-exact from the source except the tensors",
            "the operations re-initialize. The new file gets a fresh ModelID, parent_model_id = the",
            "source's ModelID, and a derivation_history record listing every operation applied.",
            "Changes that would alter any tensor's name or shape are refused. Never overwrites.",
            "",
            "  \(fromFlag) <path>      source model (.safetensors model file; not a trainer-state file)",
            "  \(outFlag) <path>       destination; must end in .safetensors and must not exist",
            "  \(groupFlag) <index>    0-based block group an operation applies to (repeatable);",
            "                   omitted = every group the operation applies to",
            "  \(initSeedFlag) <u64>   init seed for operations that draw weights (decimal UInt64);",
            "                   omitted = a drawn seed; either way it is recorded in derivation_history",
            "",
            "operations:",
        ]
        for kind in ModelDerivation.operationKinds {
            lines.append("  \(kind.flag) \(kind.valueSyntax)\(kind.acceptsGroupSelection ? "   (accepts \(groupFlag))" : "")")
            lines.append("      \(kind.summary)")
            lines.append("      changes: \(kind.changedArchitectureFields.joined(separator: ", "))")
            lines.append("      rewrites: \(kind.rewrittenTensorsDescription)")
        }
        lines += [
            "",
            "graft (changes the tensor layout; not combined with the operations above):",
            "  usage: DrewsChessMachine \(flag) \(fromFlag) <model.safetensors> \(graftToFlag) <preset | arch.json>",
            "             [\(graftMapFlag) <old=new,...>] [\(initSeedFlag) <u64>] \(outFlag) <new.safetensors>",
            "  \(graftToFlag) <preset | arch.json>",
            "      Makes a model of the target architecture (a built-in or saved preset name, or an",
            "      architecture JSON file). Every target tensor whose name and shape match a source",
            "      tensor is copied bit-exact; every other target tensor gets the value a fresh mint of",
            "      the target has under the init seed (new BN layers keep identity running statistics,",
            "      not recalibrated). Source tensors with no place in the target are dropped and listed.",
            "      The source may be trained: the output records the source's training step in its",
            "      derivation record but claims no training_step of its own.",
            "  \(graftMapFlag) <old=new,...>",
            "      Renames source tensors on the way in: exact names (a=b), whole prefixes when both",
            "      sides end in '.' (blocks.2.=blocks.3. moves a block), or a drop (a= ) so a target",
            "      tensor of the same name but another shape is initialized instead of refused.",
            "  \(initSeedFlag) <u64>   init seed for the initialized tensors; omitted = a drawn seed, recorded",
        ]
        return lines.joined(separator: "\n")
    }

    /// Inspects `rawArgs` for `--derive-model`; if present, parses, runs, and
    /// exits (never returns). Must be called on the main thread (it mints the
    /// ModelID, which is main-actor isolated).
    static func handleIfPresent(rawArgs: [String]) {
        guard rawArgs.contains(flag) else { return }

        func fail(_ message: String, _ code: Int32) -> Never {
            FileHandle.standardError.write(Data("error: \(flag): \(message)\n".utf8))
            Darwin.exit(code)
        }

        if rawArgs.contains(helpFlag) {
            print(helpText)
            Darwin.exit(0)
        }

        let operationFlags = Set(ModelDerivation.operationKinds.map(\.flag))
        let allowedFlags: Set<String> = Set([flag, fromFlag, outFlag, groupFlag, initSeedFlag, graftToFlag, graftMapFlag])
            .union(operationFlags)
        if let bad = rawArgs.first(where: { $0.hasPrefix("--") && !allowedFlags.contains($0) }) {
            fail("does not accept '\(bad)' (see \(flag) \(helpFlag))", 90)
        }

        /// Every value following an occurrence of `f`.
        func values(after f: String) -> [String] {
            var found: [String] = []
            for (index, argument) in rawArgs.enumerated() where argument == f {
                let valueIndex = index + 1
                guard valueIndex < rawArgs.count, !rawArgs[valueIndex].hasPrefix("--") else {
                    fail("\(f) requires a value", 91)
                }
                found.append(rawArgs[valueIndex])
            }
            return found
        }
        func single(_ f: String) -> String? {
            let found = values(after: f)
            guard found.count <= 1 else { fail("\(f) may be given only once", 91) }
            return found.first
        }

        guard let fromPath = single(fromFlag) else { fail("\(fromFlag) <model.safetensors> is required", 92) }
        guard let outPath = single(outFlag) else { fail("\(outFlag) <new.safetensors> is required", 92) }

        if let graftTarget = single(graftToFlag) {
            if let operationFlag = rawArgs.first(where: { operationFlags.contains($0) }) {
                fail("\(graftToFlag) cannot be combined with \(operationFlag); graft first, then derive the result", 94)
            }
            if rawArgs.contains(groupFlag) {
                fail("\(groupFlag) does not apply to \(graftToFlag)", 94)
            }
            runGraft(fromPath: fromPath, outPath: outPath, targetValue: graftTarget,
                     mapText: single(graftMapFlag), initSeedText: single(initSeedFlag), fail: fail)
        }
        if rawArgs.contains(graftMapFlag) {
            fail("\(graftMapFlag) requires \(graftToFlag)", 94)
        }

        let groupValues = values(after: groupFlag)
        var groupIndices: [Int] = []
        for raw in groupValues {
            guard let index = Int(raw), index >= 0 else { fail("\(groupFlag) '\(raw)' is not a 0-based group index", 93) }
            guard !groupIndices.contains(index) else { fail("\(groupFlag) \(index) given twice", 93) }
            groupIndices.append(index)
        }

        var operations: [any DeriveOperation] = []
        var anyOperationTakesGroups = false
        for kind in ModelDerivation.operationKinds {
            for value in values(after: kind.flag) {
                do {
                    operations.append(try kind.make(value, kind.acceptsGroupSelection && !groupIndices.isEmpty ? groupIndices : nil))
                } catch {
                    fail("\(kind.flag): \(error)", 93)
                }
                anyOperationTakesGroups = anyOperationTakesGroups || kind.acceptsGroupSelection
            }
        }
        guard !operations.isEmpty else {
            fail("no operation requested; available: \(ModelDerivation.operationKinds.map(\.flag).joined(separator: ", ")) (see \(flag) \(helpFlag))", 94)
        }
        if let seedText = single(initSeedFlag) {
            guard let seed = UInt64(seedText, radix: 10), !seedText.hasPrefix("+") else {
                fail("\(initSeedFlag) '\(seedText)' is not a decimal UInt64", 93)
            }
            var anyDraws = false
            operations = operations.map { operation in
                guard let seedable = operation as? any InitSeedableDeriveOperation, seedable.drawsWeights else { return operation }
                anyDraws = true
                return seedable.withInitSeed(seed)
            }
            if !anyDraws {
                fail("\(initSeedFlag) was given but no requested operation draws weights", 94)
            }
        }
        if !groupIndices.isEmpty, !anyOperationTakesGroups {
            fail("\(groupFlag) was given but no requested operation accepts it", 94)
        }

        let sourceURL = URL(fileURLWithPath: (fromPath as NSString).expandingTildeInPath)
        let outURL = URL(fileURLWithPath: (outPath as NSString).expandingTildeInPath)
        guard outURL.pathExtension.lowercased() == "safetensors" else {
            fail("\(outFlag) must name a .safetensors file (got \(outURL.lastPathComponent))", 95)
        }
        guard sourceURL.standardizedFileURL != outURL.standardizedFileURL else {
            fail("\(outFlag) must differ from \(fromFlag)", 95)
        }
        guard !FileManager.default.fileExists(atPath: outURL.path) else {
            fail("refusing to overwrite existing file \(outURL.path)", 95)
        }

        let modelID = MainActor.assumeIsolated { ModelIDMinter.mint().value }
        SessionLogger.shared.start()

        let sourceData: Data
        do {
            sourceData = try Data(contentsOf: sourceURL)
        } catch {
            SessionLogger.shared.shutdown()
            fail("cannot read \(sourceURL.path): \(error.localizedDescription)", 96)
        }

        let result: ModelDerivation.Result
        do {
            result = try ModelDerivation.derive(
                sourceData: sourceData,
                sourceName: sourceURL.lastPathComponent,
                operations: operations,
                newModelID: modelID,
                createdAtUnix: Int64(Date().timeIntervalSince1970),
                build: "\(BuildInfo.buildNumber) (\(BuildInfo.gitHash)\(BuildInfo.gitDirty ? "*" : ""))",
                invocationArguments: CommandLine.arguments)
        } catch {
            SessionLogger.shared.log("[DERIVE] refused \(sourceURL.lastPathComponent): \(error)")
            SessionLogger.shared.shutdown()
            fail("\(error)", 97)
        }
        result.sourceArchitectureFormat.logLegacyResolutions()
        SessionLogger.shared.log(RunProvenanceLine.line(record: result.lineage, seed: nil))

        do {
            try FileManager.default.createDirectory(at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            // Exclusive publish: the existence check above is advisory; this
            // makes "never overwrite" hold even against a race, and the
            // staged rename means a crash never leaves a torn model file.
            try FileSafety.publishNewFile(result.data, to: outURL)
        } catch {
            SessionLogger.shared.shutdown()
            fail("cannot write \(outURL.path): \(error.localizedDescription)", 98)
        }

        let header = "[DERIVE] \(sourceURL.lastPathComponent) (\(result.record.parentModelID)) -> "
            + "\(outURL.lastPathComponent) (\(modelID)); source sha256 \(result.record.sourceSHA256)"
        FileHandle.standardError.write(Data((header + "\n").utf8))
        SessionLogger.shared.log(header)
        for rewrite in result.rewrites {
            let line = "[DERIVE]   \(rewrite.operation): \(rewrite.tensorName) — \(rewrite.summary)"
            FileHandle.standardError.write(Data((line + "\n").utf8))
            SessionLogger.shared.log(line)
        }
        let summaryLine = "[DERIVE]   architecture: \(result.targetArchitecture.architectureSummary)"
        FileHandle.standardError.write(Data((summaryLine + "\n").utf8))
        SessionLogger.shared.log(summaryLine)
        // These operations never change a tensor's shape, so the size is the
        // source's: guidance only, never a refusal.
        let sizeGuidanceLine = ModelSizeGuidance.forThisMac(parameterCount: result.targetArchitecture.parameterCount)
            .logLine(event: "--derive-model \(outURL.lastPathComponent)")
        FileHandle.standardError.write(Data((sizeGuidanceLine + "\n").utf8))
        SessionLogger.shared.log(sizeGuidanceLine)

        // The path on stdout is the deliverable — reuse via --start-model.
        print(outURL.path)
        SessionLogger.shared.shutdown()
        Darwin.exit(0)
    }

    /// The graft path of `handleIfPresent`: resolve the target, build its
    /// fresh tensors, graft, write, exit. Runs on the main thread like the
    /// rest of the CLI (it blocks for the one network build).
    private static func runGraft(fromPath: String, outPath: String, targetValue: String, mapText: String?,
                                 initSeedText: String?, fail: (String, Int32) -> Never) -> Never {
        let map: GraftMap
        if let mapText {
            do {
                map = try GraftMap.parse(mapText)
            } catch {
                fail("\(error)", 93)
            }
        } else {
            map = .empty
        }
        let initSeed: UInt64
        let initSeedOrigin: String
        if let initSeedText {
            guard let seed = UInt64(initSeedText, radix: 10), !initSeedText.hasPrefix("+") else {
                fail("\(initSeedFlag) '\(initSeedText)' is not a decimal UInt64", 93)
            }
            initSeed = seed
            initSeedOrigin = "entered"
        } else {
            initSeed = WeightInitialization.drawnInitSeed()
            initSeedOrigin = "drawn"
        }
        let named: NamedArchitecture
        let targetLabel: String
        do {
            (named, targetLabel) = try ArchitecturePresetStore.resolve(nameOrPath: targetValue)
        } catch {
            fail("\(graftToFlag) \(targetValue): \(error)", 93)
        }

        let sourceURL = URL(fileURLWithPath: (fromPath as NSString).expandingTildeInPath)
        let outURL = URL(fileURLWithPath: (outPath as NSString).expandingTildeInPath)
        guard outURL.pathExtension.lowercased() == "safetensors" else {
            fail("\(outFlag) must name a .safetensors file (got \(outURL.lastPathComponent))", 95)
        }
        guard sourceURL.standardizedFileURL != outURL.standardizedFileURL else {
            fail("\(outFlag) must differ from \(fromFlag)", 95)
        }
        guard !FileManager.default.fileExists(atPath: outURL.path) else {
            fail("refusing to overwrite existing file \(outURL.path)", 95)
        }

        let modelID = MainActor.assumeIsolated { ModelIDMinter.mint().value }
        SessionLogger.shared.start()
        let seedLine = "[DERIVE] graft onto \(targetLabel): init_seed=\(initSeed) (\(initSeedOrigin)) "
            + "init_scheme=\(WeightInitScheme.current)"
        FileHandle.standardError.write(Data((seedLine + "\n").utf8))
        SessionLogger.shared.log(seedLine)
        // A graft builds the target fresh, so its size is checked here as a
        // new model's is: guidance logged, and a training state larger than
        // physical memory refused before the build.
        let sizeGuidance = ModelSizeGuidance.forThisMac(parameterCount: named.architecture.parameterCount)
        let sizeGuidanceLine = sizeGuidance.logLine(event: "--derive-model graft onto \(targetLabel)")
        FileHandle.standardError.write(Data((sizeGuidanceLine + "\n").utf8))
        SessionLogger.shared.log(sizeGuidanceLine)
        do {
            try sizeGuidance.requireTrainingStateFitsInPhysicalMemory()
        } catch {
            SessionLogger.shared.log("[DERIVE] graft refused onto \(targetLabel): \(error)")
            SessionLogger.shared.shutdown()
            fail("\(graftToFlag) \(targetValue): \(error)", 97)
        }

        let sourceData: Data
        do {
            sourceData = try Data(contentsOf: sourceURL)
        } catch {
            SessionLogger.shared.shutdown()
            fail("cannot read \(sourceURL.path): \(error.localizedDescription)", 96)
        }

        let result: ModelDerivation.GraftResult
        do {
            let fresh = try GraftFreshTarget.build(architecture: named.architecture, initSeed: initSeed)
            result = try ModelDerivation.graft(
                sourceData: sourceData,
                sourceName: sourceURL.lastPathComponent,
                fresh: fresh,
                targetLabel: targetLabel,
                map: map,
                initSeedOrigin: initSeedOrigin,
                newModelID: modelID,
                createdAtUnix: Int64(Date().timeIntervalSince1970),
                build: "\(BuildInfo.buildNumber) (\(BuildInfo.gitHash)\(BuildInfo.gitDirty ? "*" : ""))",
                invocationArguments: CommandLine.arguments)
        } catch {
            SessionLogger.shared.log("[DERIVE] graft refused \(sourceURL.lastPathComponent): \(error)")
            SessionLogger.shared.shutdown()
            fail("\(error)", 97)
        }
        result.sourceArchitectureFormat.logLegacyResolutions()
        SessionLogger.shared.log(RunProvenanceLine.line(record: result.lineage, seed: nil))

        do {
            try FileManager.default.createDirectory(at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            // Exclusive publish, as for every derive: never overwrites, never
            // leaves a torn file.
            try FileSafety.publishNewFile(result.data, to: outURL)
        } catch {
            SessionLogger.shared.shutdown()
            fail("cannot write \(outURL.path): \(error.localizedDescription)", 98)
        }

        var lines = [
            "[DERIVE] \(sourceURL.lastPathComponent) (\(result.record.parentModelID)) -> \(outURL.lastPathComponent) "
                + "(\(modelID)) by graft onto \(targetLabel); source sha256 \(result.record.sourceSHA256)",
            "[DERIVE]   copied \(result.copied.count), initialized \(result.initialized.count), "
                + "dropped \(result.dropped.count)",
        ]
        lines += result.initialized.map { "[DERIVE]   initialized \($0)" }
        lines += result.dropped.map { "[DERIVE]   dropped \($0)" }
        lines.append("[DERIVE]   architecture: \(result.targetArchitecture.architectureSummary)")
        for line in lines {
            FileHandle.standardError.write(Data((line + "\n").utf8))
            SessionLogger.shared.log(line)
        }
        // The path on stdout is the deliverable — reuse via --start-model.
        print(outURL.path)
        SessionLogger.shared.shutdown()
        Darwin.exit(0)
    }
}
