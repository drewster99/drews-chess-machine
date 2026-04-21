# `.dcmmodel` File Format

Binary serialization format for one `ChessNetwork` worth of persistent
state (trainable weights + BN running statistics). Produced by
`ModelCheckpointFile.encode()` and consumed by
`ModelCheckpointFile.decode(_:)` in
`DrewsChessMachine/DrewsChessMachine/ModelCheckpointFile.swift`.

## Design goals

- **Byte-exact round-trip.** A save followed by a load must reproduce
  every weight bit for bit. No lossy transcoding, no byte-reorder
  surprises, no float normalization.
- **Fail loudly on corruption or architecture mismatch.** The decoder
  refuses files that do not SHA-match, files minted against a different
  architecture, files with out-of-order tensors, files with implausible
  tensor sizes, and files with trailing garbage. No partial loads, no
  migration fallbacks.
- **Flat and self-describing.** No external schema. No compression. One
  `open`, one `read`, one SHA pass, done. Deliberately unclever so the
  format survives rewrites of surrounding code unchanged.
- **Apple-silicon-native byte order.** All multi-byte integers are
  little-endian. All floats are IEEE-754 binary32 in native (little-
  endian) order. No endian conversion on read or write.

## On-disk layout

All offsets are in bytes from the start of the file. All multi-byte
integers are little-endian. All floats are IEEE-754 Float32 in native
byte order.

```
Offset  Size    Field                   Type       Notes
------  ------  ---------------------   ---------  ------------------------------------
  0       8    magic                   [UInt8]    ASCII "DCMMODEL"
  8       4    formatVersion           UInt32     Currently 1
 12       4    archHash                UInt32     FNV-1a over architecture constants
 16       4    numTensors              UInt32     Total persistent-variable count
 20       8    createdAtUnix           Int64      Mint time, seconds since 1970
 28       4    modelIDByteCount (M)    UInt32     UTF-8 byte length of modelID
 32       M    modelID                 UTF-8      e.g. "20260420-3-A7F2"
 32+M     4    metadataJSONByteCount   UInt32     UTF-8 byte length of metadata JSON
 36+M     J    metadataJSON            UTF-8      sorted-keys JSON, see below

  --- repeated `numTensors` times, in declared order ---
  ..      4    tensorIndex             UInt32     Must equal its position (0-indexed)
  ..      4    elementCount            UInt32     Number of Float32 elements in this tensor
  ..      4×N  elements                Float32×N  Raw native-byte-order floats

  --- trailer ---
END-32   32    sha256                  [UInt8]    SHA-256 of all preceding bytes
```

The minimum legal encoded size is **64 bytes**: magic(8) + version(4)
+ archHash(4) + numTensors(4) + createdAt(8) + idLen(4) + zero-length
id + metaLen(4) + zero-length metadata + sha256(32).

## Field definitions

### `magic` — 8 bytes

The literal ASCII string `"DCMMODEL"`. First check the decoder
performs after the SHA-256 validates. Catches accidental feeding of
wrong-format files (e.g., a replay buffer or a PNG) before any later
field is parsed.

### `formatVersion` — UInt32

Format revision identifier. Currently **1** — there is no v0 and no
migration path. If the on-disk layout ever needs to change, the
writer bumps this, readers hard-reject all prior versions with
`ModelCheckpointError.unsupportedVersion`. Per project convention:
no backward-compatible readers. Older files stop loading.

### `archHash` — UInt32

Non-cryptographic fingerprint of the `ChessNetwork` architecture
constants that determine the variable layout:

- `ChessNetwork.channels`
- `ChessNetwork.numBlocks`
- `ChessNetwork.inputPlanes`
- `ChessNetwork.boardSize`
- `ChessNetwork.policySize`

Any change to any of these produces a different `archHash`. A file
minted with one architecture cannot be loaded into a build with a
different architecture — `ModelCheckpointError.archMismatch` is
thrown, including both the expected and got hash values in the
error message.

#### FNV-1a algorithm (full detail)

Fowler-Noll-Vo hash, version 1a. Created by Glenn Fowler, Landon Curt
Noll, and Phong Vo in 1991; variant 1a (XOR-before-multiply, better
avalanche) published by Noll in 2009. It is a simple, fast,
non-cryptographic hash with good uniformity for short inputs —
exactly the shape of input being hashed here (five small integers).

**Two magic constants** define the 32-bit variant:

| Constant           | Value        | Decimal    | Origin |
|--------------------|--------------|------------|--------|
| FNV-1a offset basis | `0x811C9DC5` | 2,166,136,261 | Defined by the FNV spec; historical, not rederivable |
| FNV prime           | `0x01000193` | 16,777,619    | `2²⁴ + 2⁸ + 0x93`, chosen for good distribution |

**Algorithm:**

```
function fnv1a_32(bytes):
    hash = 0x811C9DC5                    // 32-bit offset basis
    for each byte b in bytes:
        hash = hash XOR UInt32(b)        // XOR the byte into the low 8 bits
        hash = hash &* 0x01000193        // wrapping multiply by FNV prime
    return hash
```

The "1a" in the name indicates the **XOR-before-multiply** order,
which yields better avalanche behavior than the original FNV-1 (which
multiplies first, then XORs). Both variants use the same two
constants.

**Swift reference implementation** (from
`ModelCheckpointFile.currentArchHash`):

```swift
var h: UInt32 = 0x811C9DC5              // offset basis
func mix(_ value: Int) {
    guard let u32 = UInt32(exactly: value) else {
        preconditionFailure("Architecture constant exceeds UInt32")
    }
    var v = u32.littleEndian              // serialize as 4 LE bytes
    withUnsafeBytes(of: &v) { raw in
        for byte in raw {
            h ^= UInt32(byte)             // XOR in one byte
            h = h &* 0x01000193           // wrapping multiply
        }
    }
}
mix(ChessNetwork.channels)
mix(ChessNetwork.numBlocks)
mix(ChessNetwork.inputPlanes)
mix(ChessNetwork.boardSize)
mix(ChessNetwork.policySize)
return h
```

**Byte-order note:** each `Int` constant is serialized little-endian
(via `.littleEndian` byte reinterpretation) before being fed through
the hash byte by byte. This makes the hash value architecture-
independent for the same tuple of constants. Apple platforms are all
little-endian natively, so the explicit `.littleEndian` is
documentation as much as it is conversion — but it nails down the
spec so any future port (x86, ARM big-endian variants, cross-language
re-implementation) computes the same `archHash` for the same
`ChessNetwork` shape.

**Worked example.** For the current architecture:
- channels = 128
- numBlocks = 8
- inputPlanes = 20
- boardSize = 8
- policySize = 4864

Each value is serialized as 4 little-endian bytes, then 20 total
bytes are fed through the FNV-1a loop. The result is a specific
32-bit value that changes the instant any one of those five
constants changes. (Running `print(ModelCheckpointFile.currentArchHash)`
at launch prints the current value; it appears in the `[APP]` log
banner on startup.)

**Why not CRC32 / xxHash / SHA?**
- **CRC32** is widely available but designed for error detection on
  serial lines, not short integer tuples; it has known poor avalanche
  for small inputs.
- **xxHash / CityHash / FarmHash** are faster than FNV-1a on long
  inputs but are orders of magnitude overkill for 20 bytes, and they
  pull in more surface area than the two-constant FNV loop.
- **SHA-256** is the right choice for the file-integrity trailer
  (where cryptographic collision resistance matters) but is overkill
  for a short identifier and would balloon the `archHash` field from
  4 bytes to 32.

**Non-cryptographic by design.** Collisions are findable (a motivated
attacker could craft two architecture tuples that hash identically),
but the threat model here is accidental corruption and honest
version-skew — not adversarial tampering. The real integrity check
for the whole file is the trailing SHA-256 (see `sha256` field
below); `archHash` is specifically the short "does this file's
variable layout match the running build?" fingerprint.

### `numTensors` — UInt32

Number of persistent-state tensors following the metadata block.
Matches `ChessNetwork.trainableVariables.count +
bnRunningStatsVariables.count` at save time. If the loaded count
does not match the live network's count at restore time,
`ChessNetwork.loadWeights` surfaces the mismatch with the precise
expected/got counts.

### `createdAtUnix` — Int64

Unix timestamp (seconds since 1970-01-01 UTC) captured at
`ModelCheckpointFile` construction. Not used for format validation;
purely for the human-browsable audit trail.

### `modelID` — UTF-8, length-prefixed

Canonical model identifier minted by `ModelID` (format:
`yyyymmdd-N-XXXX`). Preserved exactly across save/load so every
stats-line reference remains traceable after a session resume. UTF-8
decode failures throw `ModelCheckpointError.invalidUTF8`.

### `metadataJSON` — UTF-8 JSON, length-prefixed

JSON-encoded `ModelCheckpointMetadata` struct:

```swift
struct ModelCheckpointMetadata: Codable, Equatable {
    let creator: String        // "manual" | "promote" | "session-autosave" | ...
    let trainingStep: Int?     // step count at save; nil for fresh-builds
    let parentModelID: String  // empty string if none
    let notes: String          // free-form human note
}
```

Encoded with `JSONEncoder.outputFormatting = [.sortedKeys]` so the
same struct always produces byte-identical bytes — matters for
reproducible file hashes. Decode failures throw
`ModelCheckpointError.invalidJSON`.

New fields may be added to the struct without bumping
`formatVersion` as long as they are Optional (older files decode
with new fields = nil). Required fields cannot be added without a
version bump.

### Tensor records

Repeated `numTensors` times, in the order declared by
`ChessNetwork.persistentVariables` (trainables first, BN running
stats second). Each record:

```
UInt32 tensorIndex      // must equal its position (0, 1, 2, ...)
UInt32 elementCount     // must be ≤ maxTensorElementCount
Float32 × elementCount  // raw little-endian IEEE-754 bytes
```

**Sanity caps** checked at decode time:
- `tensorIndex == expectedPosition` or throw
  `ModelCheckpointError.tensorIndexMismatch`. Catches corrupted or
  shuffled files immediately.
- `0 <= elementCount <= maxTensorElementCount` (computed from the
  live architecture constants, currently the largest-layer size plus
  65,536 slack) or throw
  `ModelCheckpointError.implausibleTensorSize`. Defense-in-depth for
  the astronomically-unlikely case where a corrupted file's SHA
  somehow matches.
- `elementCount × sizeof(Float32)` checked for `UInt32` overflow via
  `multipliedReportingOverflow`; overflow is treated as implausible
  size.

Tensor bytes are `memcpy`d from the input into a `[Float]` buffer.
Apple platforms are little-endian and IEEE-754-conformant, so the
bytes land directly — no per-element byteswap or normalization.

### `sha256` — 32 bytes, trailer

SHA-256 digest computed over **every preceding byte** of the file
(magic through the last tensor element). Verified **before** any
header field is parsed:

```
contentEnd = data.count - 32
content    = data[0..<contentEnd]
storedHash = data[contentEnd..<data.count]
computed   = SHA256(content)
if storedHash != computed { throw hashMismatch }
```

The SHA-first ordering means:
- Truncated files fail the SHA before header parsing can read
  garbage off the end.
- Corrupted metadata does not produce a confusing JSON error —
  SHA fires first.
- Hand-editing a field without recomputing the SHA fails loudly.
- Decode code downstream of the SHA check can trust that
  multi-byte integer reads won't walk past EOF.

At ~2 M params × 4 bytes = ~10 MB, SHA-256 costs ~30 ms on Apple
silicon — trivial relative to weight-transfer costs. Run on every
save (inside `encode`) and every load.

## Decode protocol (strict ordering)

The decoder enforces this order. Any failure at any step aborts
with a specific error:

1. `data.count >= minimumEncodedSize` → `.fileTooShort`
2. `SHA-256(data[..-32]) == data[-32..]` → `.hashMismatch`
3. `magic == "DCMMODEL"` → `.magicMismatch`
4. `formatVersion == 1` → `.unsupportedVersion(v)`
5. `archHash == currentArchHash` → `.archMismatch(expected, got)`
6. Read `numTensors`, `createdAtUnix`, `modelID`, `metadataJSON`
7. For each `expectedIndex` in `0..<numTensors`:
   - `tensorIndex == expectedIndex` → `.tensorIndexMismatch`
   - `0 <= elementCount <= maxTensorElementCount` →
     `.implausibleTensorSize`
   - `elementCount * 4` does not overflow → `.implausibleTensorSize`
   - Read `elementCount * 4` bytes (or throw `.truncated`)
8. `remaining bytes == 0` → `.trailingBytesAfterPayload(n)`
9. Return the decoded struct

## Write protocol

1. Build `Data` in-memory from header + tensors.
2. Compute `SHA-256(data)` and append the 32-byte digest.
3. Hand the `Data` to the caller; `CheckpointManager.saveModel` or
   `saveSession` writes it to a `.tmp` path with
   `Data.write(to:options: [.atomic])` for atomic rename.
4. Call `verifyModelFile(at:expectedWeights:)` — re-reads the tmp
   file, decodes it, byte-compares every tensor element against the
   in-memory weights, then runs a forward-pass round-trip through a
   scratch `ChessMPSNetwork` to catch `loadWeights`/`exportWeights`
   regressions that leave MPS state subtly wrong but whose on-disk
   bytes look right.
5. Only after verification passes does `moveItem(tmpURL, finalURL)`
   commit the file under its real name.

The `.atomic` option plus the verify-before-rename sequence gives
"successful save means a fully-valid file exists" as a hard
invariant. Failed verifications delete the tmp file and throw.

## Size estimates

For the current architecture (20 input planes, 128 channels, 8
residual blocks with SE, 76 policy channels, BN everywhere):

- Trainables: 92 tensors
- BN running stats: 36 tensors
- Total: 128 tensors
- Weight bytes: ~9.5 MB
- Header + metadata overhead: ~500 bytes
- SHA trailer: 32 bytes
- Total file size: ~10 MB

## Error taxonomy

All format-level errors are cases of `ModelCheckpointError`
(LocalizedError-conforming, surfaces human-readable messages in the
UI and log). Full list:

| Error | When |
|---|---|
| `.fileTooShort` | `data.count < minimumEncodedSize` |
| `.magicMismatch` | First 8 bytes ≠ `"DCMMODEL"` |
| `.unsupportedVersion(v)` | `formatVersion` is not a currently supported version |
| `.archMismatch(expected, got)` | `archHash` disagrees with the running build's arch |
| `.tensorCountMismatch(expected, got)` | Decoded count disagrees with the live network's count (raised by `loadWeights`, not by `decode` itself) |
| `.tensorIndexMismatch(expected, got)` | A tensor record's `tensorIndex` is out of order |
| `.elementCountMismatch(i, expected, got)` | (Raised at load time in `ChessNetwork.loadWeights` when a tensor's element count does not match the expected shape) |
| `.implausibleTensorSize(i, elementCount, maxAllowed)` | A tensor reports more elements than the arch's largest layer plus slack; or the byte count overflows UInt32 |
| `.truncated` | A read ran past the end of the payload (before the SHA trailer) |
| `.trailingBytesAfterPayload(remaining)` | Bytes left over between the last tensor and the SHA trailer |
| `.hashMismatch` | Stored SHA-256 does not match recomputed SHA-256 |
| `.invalidJSON(err)` | Metadata JSON decode failed |
| `.invalidUTF8` | modelID or metadata JSON bytes are not valid UTF-8 |
| `.encodingFailed(detail)` | Encoder failure while writing a field |

## Non-goals

- **No compression.** Weights are already dense Float32; entropy is
  high; zlib/LZ4 would save ~5-10% at the cost of build complexity
  and a variable-size payload that complicates the sanity caps.
- **No partial loading.** All-or-nothing decode.
- **No in-place editing.** Every save is a fresh file with a fresh
  timestamped name under `~/Library/Application Support/
  DrewsChessMachine/{Models,Sessions}/`. Nothing on disk is ever
  overwritten.
- **No cross-architecture migration.** If `archHash` disagrees, the
  file will not load, full stop.
- **No backward compatibility beyond the current format version.**
  Old versions load-fail with a precise error.
