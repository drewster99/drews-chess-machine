//
//  NewModelCLI.swift
//
//  Headless `--new-model` pre-flight: mint a FRESH, untrained network from a
//  named preset and write it to a `.safetensors` file, then exit — no training,
//  no GUI. The point is a single, fixed starting net that can be reused as
//  `--start-model` across multiple runs (e.g. clean A/B comparisons where every
//  run must begin from byte-identical weights).
//
//  Cost profile mirrors `--probe-model`: one network build (`.randomWeights`,
//  which includes a one-shot BN warmup forward) + a weight export + a file
//  write. Forward-only, so it coexists with a running training job the same way
//  a probe does (it does NOT open a second training command stream).
//

import Foundation

enum NewModelCLI {

    /// Build `preset`'s architecture with random weights, write the safetensors
    /// to `outPath` (or a default under the Models dir), print the path, exit.
    /// `modelID` is minted by the caller on the main actor (the minter is
    /// main-actor isolated; this routine runs its GPU work off-actor).
    static func runAndExit(presetName: String, outPath: String?, modelID: String) -> Never {
        SessionLogger.shared.start()

        guard let preset = NetworkArchitecture.Preset(rawValue: presetName) else {
            let valid = NetworkArchitecture.Preset.allCases.map(\.rawValue).joined(separator: ", ")
            FileHandle.standardError.write(Data(
                "error: --new-model: unknown preset '\(presetName)'. Valid: \(valid)\n".utf8
            ))
            Darwin.exit(70)
        }

        let arch = NetworkArchitecture.preset(preset)
        do {
            try arch.validate()
        } catch {
            FileHandle.standardError.write(Data(
                "error: --new-model: preset '\(presetName)' failed validation: \(error)\n".utf8
            ))
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
                .appendingPathComponent("\(presetName)-fresh-\(modelID).safetensors")
        }

        // Never overwrite — same discipline as CheckpointManager.saveModel. A
        // reusable starting net is precious; refuse rather than stomp it.
        if FileManager.default.fileExists(atPath: outURL.path) {
            FileHandle.standardError.write(Data(
                "error: --new-model: refusing to overwrite existing file \(outURL.path)\n".utf8
            ))
            Darwin.exit(72)
        }

        FileHandle.standardError.write(Data(
            "[NEW-MODEL] minting \(presetName) (v\(arch.architectureVersionLabel), \(arch.parameterCount) params) id=\(modelID)\n".utf8
        ))

        do {
            // Build with random weights (includes the BN warmup forward) and
            // export the persistent tensors, off the main actor.
            let weights = try syncWait { () async throws -> [[Float]] in
                let net = try ChessMPSNetwork(.randomWeights, arch: arch)
                return try await net.network.exportWeights()
            }
            // Sanity: the exported tensor count must equal the plan — the same
            // index-aligned contract the loaders rely on.
            let planCount = arch.weightTensorPlan().count
            guard weights.count == planCount else {
                FileHandle.standardError.write(Data(
                    "error: --new-model: exported \(weights.count) tensors but plan expects \(planCount)\n".utf8
                ))
                Darwin.exit(73)
            }

            let metadata = ModelCheckpointMetadata(
                creator: "new-model",
                trainingStep: nil,
                parentModelID: "",
                notes: "fresh \(presetName) net (untrained), arch v\(arch.architectureVersionLabel)"
            )
            let encoded = try SafetensorsModelIO.encode(
                modelID: modelID,
                createdAtUnix: Int64(Date().timeIntervalSince1970),
                metadata: metadata,
                weights: weights,
                architecture: arch,
                includesVelocity: false
            )
            try FileManager.default.createDirectory(
                at: outURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try encoded.write(to: outURL, options: [.atomic])
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
