//
//  NewModelCLI.swift
//
//  Headless `--new-model` pre-flight: mint a FRESH, untrained network from a
//  named preset and write it to a `.safetensors` file, then exit — no training,
//  no GUI. The point is a single, fixed starting net that can be reused as
//  `--start-model` across multiple runs (e.g. clean A/B comparisons where every
//  run must begin from byte-identical weights).
//
//  Cost profile mirrors `--probe-model`: one network build (a seeded fresh
//  network, which includes a one-shot BN warmup forward) + a weight export + a
//  file write. The init seed (`--init-seed`, or drawn) and its scheme are
//  logged and written into the file, so the same seed re-mints bit-identical
//  trainable tensors; the BN running statistics come from the GPU warmup
//  forward and match only to float tolerance (across chips or OS builds, and
//  for a bf16/fp16 model across policy tail precisions, their last bits can
//  differ; the architecture records the tail, and the notes repeat it). Forward-only, so it coexists with a running training job the same way
//  a probe does (it does NOT open a second training command stream).
//

import Foundation

enum NewModelCLI {

    /// Build `arch` with random weights, write the safetensors to `outPath`
    /// (or a default under the Models dir), print the path, exit. `name` is a
    /// short source label (the built-in preset name, the user-saved preset
    /// name, or an arch-file stem) used for the default filename and logs; the
    /// caller has already resolved + validated `arch` (built-in preset or JSON
    /// preset/file via `ArchitecturePresetStore`). `modelID` is minted by the
    /// caller on the main actor (the minter is main-actor isolated; this
    /// routine runs its GPU work off-actor). Logs the `[ARCH] size guidance`
    /// line for this Mac (`ModelSizeGuidance`) and refuses, like a failed
    /// validation, a model whose training state cannot fit in physical
    /// memory. `naming` (`--name` and the preset `--architecture` named) is
    /// recorded in the file's lineage.
    static func runAndExit(architecture arch: NetworkArchitecture, name: String, naming: ModelNaming, outPath: String?,
                           modelID: String, enteredInitSeed: UInt64?) -> Never {
        SessionLogger.shared.start()

        // Defensive re-validation (the caller already validated via the store).
        do {
            try arch.validate()
        } catch {
            FileHandle.standardError.write(Data(
                "error: --new-model: architecture '\(name)' failed validation: \(error)\n".utf8
            ))
            Darwin.exit(71)
        }

        // Size guidance for this Mac (`ModelSizeGuidance`): logged for every
        // mint, and the one size refused — a training state larger than
        // physical memory, which no run here could hold.
        let sizeGuidance = ModelSizeGuidance.forThisMac(parameterCount: arch.parameterCount)
        let sizeGuidanceLine = sizeGuidance.logLine(event: "--new-model \(name)")
        FileHandle.standardError.write(Data((sizeGuidanceLine + "\n").utf8))
        SessionLogger.shared.log(sizeGuidanceLine)
        do {
            try sizeGuidance.requireTrainingStateFitsInPhysicalMemory()
        } catch {
            FileHandle.standardError.write(Data(
                "error: --new-model: architecture '\(name)' \(error)\n".utf8
            ))
            SessionLogger.shared.shutdown()
            Darwin.exit(71)
        }

        // Resolve the destination BEFORE the (slower) build so a bad path fails
        // fast. Default lands in the curated Models dir with a name that records
        // the preset and the minted ID; an explicit --out-model wins and is
        // normalized to a `.safetensors` extension (the loaders key off it).
        let outURL: URL
        if let outPath {
            let expanded = (outPath as NSString).expandingTildeInPath
            let u = URL(fileURLWithPath: expanded)
            outURL = u.pathExtension.lowercased() == "safetensors"
                ? u
                : u.appendingPathExtension("safetensors")
        } else {
            outURL = CheckpointPaths.modelsDir
                .appendingPathComponent("\(name)-fresh-\(modelID).safetensors")
        }

        // Never overwrite — same discipline as CheckpointManager.saveModel. A
        // reusable starting net is precious; refuse rather than stomp it.
        // This check only fails fast, before the slow build; the guarantee is
        // the exclusive publish below, which holds even when something
        // appears at the path while the network is being built.
        if FileManager.default.fileExists(atPath: outURL.path) {
            FileHandle.standardError.write(Data(
                "error: --new-model: refusing to overwrite existing file \(outURL.path)\n".utf8
            ))
            Darwin.exit(72)
        }

        let initSeed: UInt64
        let initSeedSource: String
        if let enteredInitSeed {
            initSeed = enteredInitSeed
            initSeedSource = "entered"
        } else {
            initSeed = WeightInitialization.drawnInitSeed()
            initSeedSource = "drawn"
        }
        let initLine = "init_seed=\(initSeed) (\(initSeedSource)) init_scheme=\(WeightInitScheme.current)"
        FileHandle.standardError.write(Data(
            "[NEW-MODEL] minting \(name) (\(arch.parameterCount) params) id=\(modelID) \(ModelNaming.logText(.recorded(naming))) \(initLine)\n".utf8
        ))
        SessionLogger.shared.log("[NEW-MODEL] minting \(name) id=\(modelID) \(ModelNaming.logText(.recorded(naming))) \(initLine)")

        do {
            // Build with random weights (includes the BN warmup forward) and
            // export the persistent tensors, off the main actor.
            let (weights, buildTimeMs) = try syncWait { () async throws -> ([[Float]], Double) in
                let net = try ChessMPSNetwork(.randomWeights(initSeed: initSeed), arch: arch)
                return (try await net.network.exportWeights(), net.buildTimeMs)
            }
            let builtLine = "[NEW-MODEL] built \(name) in \(String(format: "%.1f", buildTimeMs)) ms"
            FileHandle.standardError.write(Data((builtLine + "\n").utf8))
            SessionLogger.shared.log(builtLine)
            // Sanity: the exported tensor count must equal the plan — the same
            // index-aligned contract the loaders rely on.
            let planCount = arch.weightTensorPlan().count
            guard weights.count == planCount else {
                FileHandle.standardError.write(Data(
                    "error: --new-model: exported \(weights.count) tensors but plan expects \(planCount)\n".utf8
                ))
                Darwin.exit(73)
            }

            let initialization = ModelInitRecord(initSeed: initSeed, scheme: WeightInitScheme.current)
            let metadata = ModelCheckpointMetadata(
                creator: "new-model",
                trainingStep: nil,
                parentModelID: "",
                notes: "fresh \(name) net (untrained), "
                    + "BN warm-up under policy tail precision \(arch.policyTailPrecision.rawValue)"
            )
            let mintDate = Date()
            let testSetResults = try syncWait { () async throws -> ModelTestSetResultsField in
                await ModelTestSetEvaluator.modelFiles.evaluateForSave(weights: weights, architecture: arch, file: outURL.lastPathComponent)
            }
            let encoded = try SafetensorsModelIO.encode(
                modelID: modelID,
                createdAtUnix: Int64(mintDate.timeIntervalSince1970),
                metadata: metadata,
                weights: weights,
                architecture: arch,
                includesVelocity: false,
                lineage: try LineageTracker.mintRecord(pathKind: .newModel, argv: CommandLine.arguments,
                                                       initialization: initialization, naming: naming, at: mintDate),
                testSetResults: testSetResults
            )
            try FileManager.default.createDirectory(
                at: outURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            // Staged in a temporary sibling and renamed into place only if
            // nothing — file, folder or symbolic link — is at the path, so
            // neither something that got there after the check above nor a
            // half-written file under the final name is possible.
            do {
                try FileSafety.publishNewFile(encoded, to: outURL)
            } catch FileSafetyError.alreadyExists(path: _, kind: let kind) {
                FileHandle.standardError.write(Data(
                    "error: --new-model: refusing to overwrite \(outURL.path): a \(kind) already exists at that path (created while the network was being built, or a symbolic link)\n".utf8
                ))
                SessionLogger.shared.shutdown()
                Darwin.exit(72)
            }
        } catch {
            FileHandle.standardError.write(Data(
                "error: --new-model: build/save failed: \(error)\n".utf8
            ))
            SessionLogger.shared.shutdown()
            Darwin.exit(74)
        }

        // The path on stdout is the deliverable — copy it into --start-model.
        print(outURL.path)
        FileHandle.standardError.write(Data(
            "[NEW-MODEL] wrote \(outURL.lastPathComponent) — reuse via: --start-model \(outURL.path)\n".utf8
        ))
        SessionLogger.shared.shutdown()
        Darwin.exit(0)
    }

    private static func syncWait<T>(_ work: @Sendable @escaping () async throws -> T) throws -> T {
        let box = NewModelSyncBox<T>()
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            do { box.success = try await work() }
            catch { box.failure = error }
            semaphore.signal()
        }
        semaphore.wait()
        if let error = box.failure { throw error }
        guard let success = box.success else {
            preconditionFailure("NewModelCLI.syncWait: result box carried neither success nor failure")
        }
        return success
    }
}

private final class NewModelSyncBox<T>: @unchecked Sendable {
    var success: T?
    var failure: Error?
}
