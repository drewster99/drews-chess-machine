import Foundation

/// On-disk format for the optional `training_chart.json` and
/// `progress_rate_chart.json` companion files inside a
/// `.dcmsession` bundle. Plain JSON, not compressed — at ~30–50 MB
/// per 24h session these are noise next to `replay_buffer.bin` in
/// the same bundle, and plain JSON is inspectable directly with
/// `jq`. Codable also handles schema evolution gracefully (new
/// optional fields decode as nil on older saved files), which
/// matches how chart-sample structs evolve in practice.
///
/// Each file holds a single JSON object — not a bare array — so
/// forward-compat metadata travels alongside the samples:
///
///     {
///       "formatVersion": 1,
///       "sampleCount": 12453,
///       "samples": [ {...}, {...}, ... ]
///     }
///
/// `formatVersion` lets the loader reject a future incompatible
/// schema gracefully; `sampleCount` lets the loader catch
/// truncation by cross-checking against `samples.count`.
///
/// `Double.nan` and `Double.infinity` are not valid in RFC 8259
/// JSON — `JSONEncoder` would otherwise throw. Both encoder and
/// decoder are configured with the matching
/// `nonConformingFloatEncodingStrategy` so legitimate NaN values
/// (e.g. `gNorm` after a divergence) round-trip as the strings
/// `"NaN"`, `"+Inf"`, `"-Inf"`.

enum ChartFileError: LocalizedError {
    case unsupportedFormatVersion(Int)
    case sampleCountMismatch(declared: Int, actual: Int)
    case readFailed(URL, Error)
    case writeFailed(URL, Error)
    case decodeFailed(URL, Error)

    var errorDescription: String? {
        switch self {
        case .unsupportedFormatVersion(let v):
            return "Unsupported chart-file formatVersion: \(v)"
        case .sampleCountMismatch(let declared, let actual):
            return "Chart-file sampleCount mismatch: declared \(declared), found \(actual) samples"
        case .readFailed(let url, let err):
            return "Could not read \(url.lastPathComponent): \(err.localizedDescription)"
        case .writeFailed(let url, let err):
            return "Could not write \(url.lastPathComponent): \(err.localizedDescription)"
        case .decodeFailed(let url, let err):
            return "Could not decode \(url.lastPathComponent): \(err.localizedDescription)"
        }
    }
}

/// Current chart-file format version. Bump when making a breaking
/// change to the on-disk envelope. Additive `Optional` fields on
/// the underlying sample structs do NOT require a bump.
let chartFileCurrentFormatVersion: Int = 1

private struct ChartFileEnvelope<S: Codable>: Codable {
    let formatVersion: Int
    let sampleCount: Int
    let samples: [S]
}

private func makeChartEncoder() -> JSONEncoder {
    let encoder = JSONEncoder()
    // .convertToString lets Double.nan / ±Infinity round-trip as
    // explicit strings ("NaN" / "+Inf" / "-Inf"). Without this the
    // encoder throws on NaN, and gNorm legitimately becomes NaN
    // after a divergence — we want the chart to record that.
    encoder.nonConformingFloatEncodingStrategy = .convertToString(
        positiveInfinity: "+Inf",
        negativeInfinity: "-Inf",
        nan: "NaN"
    )
    return encoder
}

private func makeChartDecoder() -> JSONDecoder {
    let decoder = JSONDecoder()
    decoder.nonConformingFloatDecodingStrategy = .convertFromString(
        positiveInfinity: "+Inf",
        negativeInfinity: "-Inf",
        nan: "NaN"
    )
    return decoder
}

/// Encode `samples` into the standard chart-file envelope and
/// write it atomically to `url`. Caller is responsible for any
/// fsync / parent-directory flush — this matches how the existing
/// `replay_buffer.bin` write path layers durability on top of a
/// raw write.
func writeChartFile<S: Codable>(
    _ samples: [S],
    formatVersion: Int = chartFileCurrentFormatVersion,
    to url: URL
) throws {
    let envelope = ChartFileEnvelope(
        formatVersion: formatVersion,
        sampleCount: samples.count,
        samples: samples
    )
    let data: Data
    do {
        data = try makeChartEncoder().encode(envelope)
    } catch {
        throw ChartFileError.writeFailed(url, error)
    }
    do {
        try data.write(to: url, options: [.atomic])
    } catch {
        throw ChartFileError.writeFailed(url, error)
    }
}

/// Read and decode a chart file written by `writeChartFile`.
/// Throws `ChartFileError.unsupportedFormatVersion` for an
/// unrecognized version, and `.sampleCountMismatch` if the
/// declared count disagrees with the deserialized array length
/// (catches truncation a bare JSON-decode wouldn't surface).
func readChartFile<S: Codable>(
    _ type: [S].Type,
    from url: URL
) throws -> [S] {
    let data: Data
    do {
        data = try Data(contentsOf: url)
    } catch {
        throw ChartFileError.readFailed(url, error)
    }
    let envelope: ChartFileEnvelope<S>
    do {
        envelope = try makeChartDecoder().decode(ChartFileEnvelope<S>.self, from: data)
    } catch {
        throw ChartFileError.decodeFailed(url, error)
    }
    guard envelope.formatVersion == chartFileCurrentFormatVersion else {
        throw ChartFileError.unsupportedFormatVersion(envelope.formatVersion)
    }
    guard envelope.sampleCount == envelope.samples.count else {
        throw ChartFileError.sampleCountMismatch(
            declared: envelope.sampleCount,
            actual: envelope.samples.count
        )
    }
    return envelope.samples
}
