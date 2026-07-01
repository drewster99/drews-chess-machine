# ML Training Lineage, Runs, Sessions, Checkpoints, and Evaluation

## Core terminology

### ModelVersion

A `ModelVersion` is a specific set of weights/state at a point in time.

Examples:

- `M0`: random initialization
- `MA`: result of training `M0` on corpus A
- `MB`: result of training `M0` on corpus B
- `MC`: result of training `MB` on corpus C
- `MD`: result of training `MB` on corpus D

### TrainingRun

A `TrainingRun` is a transformation from one `ModelVersion` into another.

```text
M0 --RunA--> MA
M0 --RunB--> MB
MB --RunC--> MC
MB --RunD--> MD
```

### TrainingSession

A `TrainingSession` is one uninterrupted execution chunk of a `TrainingRun`.

If a run stops and resumes twice, it has three sessions.

### Checkpoint

A `Checkpoint` is a saved state/artifact at a specific step/time.

### EvaluationRun

An `EvaluationRun` is a probe/eval suite executed against a model/checkpoint.

### MetricPoint

A `MetricPoint` is one scalar observation used for graphing.

```text
eval.pElo   = 1412 at step 64000
eval.nll    = 2.24 at step 64000
train.loss  = 1.83 at step 64001
```

---

## Recommended hierarchy

```text
ModelLineage
  ModelVersion
  TrainingRun
    TrainingSession
    Checkpoint
      EvaluationRun
```

Important distinction:

```text
ModelVersion    = node in the lineage graph
TrainingRun     = edge between model versions
TrainingSession = one execution chunk of a run
Checkpoint      = saved artifact/state
EvaluationRun   = testing of a checkpoint
MetricPoint     = flat chart/query fact
```

---

## Example lineage

```text
M0
├── Run A -> MA
│   ├── Session A1
│   ├── Session A2
│   └── Session A3
│
└── Run B -> MB
    └── Session B1
        ├── Run C -> MC
        └── Run D -> MD
```

Graph-edge view:

```text
M0 --RunA--> MA
M0 --RunB--> MB
MB --RunC--> MC
MB --RunD--> MD
```

---

## Main design rule

Use two layers:

### 1. Normalized lineage model

Used for truth, reproducibility, ancestry, checkpoints, and resumability.

### 2. Flat metric/event table

Used for charting, dashboards, comparisons, and simple queries.

Do not make charts walk the lineage tree. Emit chartable values into `MetricPoint`.

---

## Swift data structures

### ModelLineage

```swift
struct ModelLineage: Identifiable, Codable {
    var id: UUID
    var name: String

    var rootModelVersionID: ModelVersion.ID

    var modelVersions: [ModelVersion.ID: ModelVersion]
    var trainingRuns: [TrainingRun.ID: TrainingRun]
    var sessions: [TrainingSession.ID: TrainingSession]
    var checkpoints: [Checkpoint.ID: Checkpoint]
    var evaluationRuns: [EvaluationRun.ID: EvaluationRun]

    var createdAt: Date
    var metadata: [String: String]
}
```

### ModelVersion

```swift
struct ModelVersion: Identifiable, Codable {
    var id: UUID

    /// Human-readable name: "M0", "MA", "MB", etc.
    var name: String

    /// Random init, checkpoint-derived, completed model, etc.
    var kind: ModelVersionKind

    /// The TrainingRun that produced this model, nil for root/random init.
    var parentTrainingRunID: TrainingRun.ID?

    /// Optional direct ancestry shortcut.
    var parentModelVersionID: ModelVersion.ID?

    /// Where the model weights/state live.
    var artifactURI: String?

    /// Hash of model weights/config for reproducibility.
    var artifactHash: String?

    /// Step within the producing run, if applicable.
    var step: Int?

    /// Elapsed training time within the producing run.
    var elapsedTrainingSeconds: Double?

    var createdAt: Date

    /// Cached summaries only; full history belongs in MetricPoint.
    var summaryMetrics: [String: Double]

    var metadata: [String: String]
}

enum ModelVersionKind: String, Codable {
    case randomInit
    case checkpoint
    case probeCheckpoint
    case completed
    case intermediate
}
```

### TrainingRun

```swift
struct TrainingRun: Identifiable, Codable {
    var id: UUID

    /// Edge: parent model -> child model.
    var parentModelVersionID: ModelVersion.ID
    var childModelVersionID: ModelVersion.ID?

    var name: String
    var corpusID: Corpus.ID?
    var datasetID: Dataset.ID?
    var configID: TrainingConfig.ID

    var status: TrainingRunStatus

    var startedAt: Date
    var completedAt: Date?

    /// These are denormalized convenience lists.
    var sessionIDs: [TrainingSession.ID]
    var checkpointIDs: [Checkpoint.ID]

    /// Useful display label for charts.
    var seriesKey: String
    var seriesLabel: String

    var metadata: [String: String]
}

enum TrainingRunStatus: String, Codable {
    case queued
    case running
    case paused
    case completed
    case failed
    case cancelled
}
```

### TrainingSession

```swift
struct TrainingSession: Identifiable, Codable {
    var id: UUID
    var trainingRunID: TrainingRun.ID

    var status: TrainingSessionStatus

    var startedAt: Date
    var endedAt: Date?

    var startStep: Int
    var endStep: Int?

    var startCheckpointID: Checkpoint.ID?
    var endCheckpointID: Checkpoint.ID?

    /// Optional runtime identity.
    var machineID: String?
    var processID: Int?

    /// Example: git SHA, executable version, Metal device name, etc.
    var environment: [String: String]
}

enum TrainingSessionStatus: String, Codable {
    case running
    case completed
    case crashed
    case stopped
}
```

### Checkpoint

```swift
struct Checkpoint: Identifiable, Codable {
    var id: UUID

    var trainingRunID: TrainingRun.ID
    var sessionID: TrainingSession.ID?

    /// ModelVersion represented by this checkpoint.
    var modelVersionID: ModelVersion.ID

    var step: Int
    var elapsedTrainingSeconds: Double
    var wallClockTime: Date

    var kind: CheckpointKind

    /// Where the checkpoint artifact lives.
    var artifactURI: String?

    /// Hash of model/optimizer/RNG state, if available.
    var artifactHash: String?

    /// False for temporary/probe checkpoints that can be deleted.
    var retained: Bool

    var metadata: [String: String]
}

enum CheckpointKind: String, Codable {
    case periodic
    case probe
    case manual
    case final
}
```

### EvaluationRun

```swift
struct EvaluationRun: Identifiable, Codable {
    var id: UUID

    /// The model/checkpoint being evaluated.
    var modelVersionID: ModelVersion.ID
    var checkpointID: Checkpoint.ID?

    var trainingRunID: TrainingRun.ID?

    /// Example: "probe-suite-v1", "validation-loss", "arena-200-games".
    var suiteName: String
    var suiteVersion: String?

    var datasetID: Dataset.ID?
    var probeSuiteID: ProbeSuite.ID?

    var status: EvaluationStatus

    var startedAt: Date
    var completedAt: Date?

    /// Cached summaries only. Full scalar history belongs in MetricPoint.
    var summaryMetrics: [String: Double]

    /// Logs, plots, sample outputs, confusion matrices, etc.
    var artifactURIs: [String: String]

    var metadata: [String: String]
}

enum EvaluationStatus: String, Codable {
    case queued
    case running
    case completed
    case failed
}
```

### MetricPoint

This is the important charting table.

```swift
struct MetricPoint: Identifiable, Codable {
    var id: UUID

    // Identity / grouping
    var lineageID: ModelLineage.ID
    var modelVersionID: ModelVersion.ID?
    var trainingRunID: TrainingRun.ID?
    var sessionID: TrainingSession.ID?
    var checkpointID: Checkpoint.ID?
    var evaluationRunID: EvaluationRun.ID?

    // X-axes
    var step: Int?
    var elapsedTrainingSeconds: Double?
    var wallClockTime: Date

    // Metric
    var metricName: String       // "train.loss", "eval.nll", "eval.pElo"
    var value: Double

    // Chart grouping / labels
    var seriesKey: String        // "v5", "mini2b", etc.
    var seriesLabel: String      // "v5 - 8.45M, 5-block ..."
    var phase: String?           // "pretrain", "finetune", "probe", etc.

    // Optional slicing dimensions
    var datasetID: Dataset.ID?
    var corpusID: Corpus.ID?
    var probeSuiteID: ProbeSuite.ID?

    var createdAt: Date
}
```

### Optional strongly-typed IDs

If you want stronger compile-time separation than raw `UUID`, use wrapper IDs.

```swift
struct ModelID<Tag>: Hashable, Codable, Sendable {
    var rawValue: UUID
}

enum ModelVersionTag {}
enum TrainingRunTag {}
enum TrainingSessionTag {}
enum CheckpointTag {}
enum EvaluationRunTag {}

typealias ModelVersionID = ModelID<ModelVersionTag>
typealias TrainingRunID = ModelID<TrainingRunTag>
typealias TrainingSessionID = ModelID<TrainingSessionTag>
typealias CheckpointID = ModelID<CheckpointTag>
typealias EvaluationRunID = ModelID<EvaluationRunTag>
```

---

## Placeholder supporting types

```swift
struct Corpus: Identifiable, Codable {
    var id: UUID
    var name: String
    var uri: String?
    var metadata: [String: String]
}

struct Dataset: Identifiable, Codable {
    var id: UUID
    var name: String
    var uri: String?
    var split: String?
    var metadata: [String: String]
}

struct TrainingConfig: Identifiable, Codable {
    var id: UUID
    var name: String
    var values: [String: String]
}

struct ProbeSuite: Identifiable, Codable {
    var id: UUID
    var name: String
    var version: String
    var probeNames: [String]
}
```

---

## Example: creating your described lineage

```swift
let lineageID = UUID()

let m0 = ModelVersion(
    id: UUID(),
    name: "M0",
    kind: .randomInit,
    parentTrainingRunID: nil,
    parentModelVersionID: nil,
    artifactURI: "models/M0/init.bin",
    artifactHash: nil,
    step: 0,
    elapsedTrainingSeconds: 0,
    createdAt: Date(),
    summaryMetrics: [:],
    metadata: [:]
)

let runA = TrainingRun(
    id: UUID(),
    parentModelVersionID: m0.id,
    childModelVersionID: nil,
    name: "Run A",
    corpusID: UUID(),
    datasetID: nil,
    configID: UUID(),
    status: .running,
    startedAt: Date(),
    completedAt: nil,
    sessionIDs: [],
    checkpointIDs: [],
    seriesKey: "corpus-a",
    seriesLabel: "Corpus A",
    metadata: [:]
)

let runB = TrainingRun(
    id: UUID(),
    parentModelVersionID: m0.id,
    childModelVersionID: nil,
    name: "Run B",
    corpusID: UUID(),
    datasetID: nil,
    configID: UUID(),
    status: .running,
    startedAt: Date(),
    completedAt: nil,
    sessionIDs: [],
    checkpointIDs: [],
    seriesKey: "corpus-b",
    seriesLabel: "Corpus B",
    metadata: [:]
)
```

After Run B completes:

```swift
let mb = ModelVersion(
    id: UUID(),
    name: "MB",
    kind: .completed,
    parentTrainingRunID: runB.id,
    parentModelVersionID: m0.id,
    artifactURI: "models/MB/final.bin",
    artifactHash: nil,
    step: 135_000,
    elapsedTrainingSeconds: 15.2 * 3600,
    createdAt: Date(),
    summaryMetrics: ["eval.nll": 2.27, "eval.pElo": 1420],
    metadata: [:]
)
```

Then fork from `MB`:

```swift
let runC = TrainingRun(
    id: UUID(),
    parentModelVersionID: mb.id,
    childModelVersionID: nil,
    name: "Run C",
    corpusID: UUID(),
    datasetID: nil,
    configID: UUID(),
    status: .queued,
    startedAt: Date(),
    completedAt: nil,
    sessionIDs: [],
    checkpointIDs: [],
    seriesKey: "corpus-c",
    seriesLabel: "Corpus C from MB",
    metadata: [:]
)

let runD = TrainingRun(
    id: UUID(),
    parentModelVersionID: mb.id,
    childModelVersionID: nil,
    name: "Run D",
    corpusID: UUID(),
    datasetID: nil,
    configID: UUID(),
    status: .queued,
    startedAt: Date(),
    completedAt: nil,
    sessionIDs: [],
    checkpointIDs: [],
    seriesKey: "corpus-d",
    seriesLabel: "Corpus D from MB",
    metadata: [:]
)
```

---

## Example: periodic probe checkpoint every 1000 steps

```swift
func makeProbeCheckpoint(
    trainingRunID: TrainingRun.ID,
    sessionID: TrainingSession.ID?,
    modelVersionID: ModelVersion.ID,
    step: Int,
    elapsedTrainingSeconds: Double,
    artifactURI: String
) -> Checkpoint {
    Checkpoint(
        id: UUID(),
        trainingRunID: trainingRunID,
        sessionID: sessionID,
        modelVersionID: modelVersionID,
        step: step,
        elapsedTrainingSeconds: elapsedTrainingSeconds,
        wallClockTime: Date(),
        kind: .probe,
        artifactURI: artifactURI,
        artifactHash: nil,
        retained: false,
        metadata: [:]
    )
}
```

---

## Example: emitting evaluation metrics for charting

```swift
func metricPoint(
    lineageID: ModelLineage.ID,
    trainingRun: TrainingRun,
    checkpoint: Checkpoint,
    evaluationRun: EvaluationRun,
    metricName: String,
    value: Double
) -> MetricPoint {
    MetricPoint(
        id: UUID(),
        lineageID: lineageID,
        modelVersionID: checkpoint.modelVersionID,
        trainingRunID: trainingRun.id,
        sessionID: checkpoint.sessionID,
        checkpointID: checkpoint.id,
        evaluationRunID: evaluationRun.id,
        step: checkpoint.step,
        elapsedTrainingSeconds: checkpoint.elapsedTrainingSeconds,
        wallClockTime: checkpoint.wallClockTime,
        metricName: metricName,
        value: value,
        seriesKey: trainingRun.seriesKey,
        seriesLabel: trainingRun.seriesLabel,
        phase: "probe",
        datasetID: trainingRun.datasetID,
        corpusID: trainingRun.corpusID,
        probeSuiteID: evaluationRun.probeSuiteID,
        createdAt: Date()
    )
}
```

Example emitted points:

```swift
let pEloPoint = metricPoint(
    lineageID: lineageID,
    trainingRun: runB,
    checkpoint: checkpoint,
    evaluationRun: evalRun,
    metricName: "eval.pElo",
    value: 1420
)

let nllPoint = metricPoint(
    lineageID: lineageID,
    trainingRun: runB,
    checkpoint: checkpoint,
    evaluationRun: evalRun,
    metricName: "eval.nll",
    value: 2.27
)
```

---

## Chart query examples

### pElo by SGD step

```sql
SELECT step, value, series_label
FROM metric_points
WHERE metric_name = 'eval.pElo'
ORDER BY series_key, step;
```

### pElo by training time

```sql
SELECT elapsed_training_seconds, value, series_label
FROM metric_points
WHERE metric_name = 'eval.pElo'
ORDER BY series_key, elapsed_training_seconds;
```

### nll by SGD step

```sql
SELECT step, value, series_label
FROM metric_points
WHERE metric_name = 'eval.nll'
ORDER BY series_key, step;
```

### nll by training time

```sql
SELECT elapsed_training_seconds, value, series_label
FROM metric_points
WHERE metric_name = 'eval.nll'
ORDER BY series_key, elapsed_training_seconds;
```

---

## Mapping to the screenshot-style charts

Each plotted dot should come from one `MetricPoint`.

For pElo:

```text
metricName = "eval.pElo"
value = 1412
step = 64000
elapsedTrainingSeconds = 54720
seriesKey = "v5"
seriesLabel = "v5 - 8.45M, 5-block (wd1e-4+wd5e-4+m0.93)"
```

For nll:

```text
metricName = "eval.nll"
value = 2.24
step = 64000
elapsedTrainingSeconds = 54720
seriesKey = "v5"
seriesLabel = "v5 - 8.45M, 5-block (wd1e-4+wd5e-4+m0.93)"
```

The charting layer should not care whether the point came from session 1, 2, or 3. That remains available for slicing/debugging, but the default chart only needs:

```text
x = step OR elapsedTrainingSeconds
y = value
series = seriesKey / seriesLabel
metricName = selected metric
```

---

## Final recommendation

Keep lineage and metrics separate.

Truth model:

```text
ModelVersion
TrainingRun
TrainingSession
Checkpoint
EvaluationRun
```

Visualization model:

```text
MetricPoint
```

This gives you:

- clean ancestry tracking
- support for branching/forks
- support for resumable runs
- support for temporary probe checkpoints
- simple plotting by step or elapsed time
- easy comparisons across architecture/config/corpus variants
