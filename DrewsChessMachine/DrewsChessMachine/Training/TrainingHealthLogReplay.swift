import Foundation

/// The offline replay of the training-health rules over saved session logs:
/// it parses `[REPLAY]` / `[VS-UCI]` step rows and `[LAYER-HEALTH]` lines and
/// feeds them to the real `TrainingHealthMonitor` and evaluator, so
/// validation never needs a second copy of the rules (the alarms plan, D4).
/// Pure: the caller reads the files (`--replay-health-log`, P2) or the test
/// bundle's excerpts and passes their text; nothing here touches a file.
///
/// **Sparse semantics** — the only differences from the app:
/// - each step row is one live evaluation whose window holds one record;
/// - the spike references are the rows in the 1,000 trainer steps before the
///   row, under the same span definition;
/// - "steps trained by this process" (rule 3's gate) is the row's own
///   `step=` (segment steps), summed over the logs passed together; a
///   corpus-replay or train-vs-UCI checkpoint headline uses its own `step=`,
///   and a dedicated `[LAYER-HEALTH] value-fc1` line its own `trained=`; a
///   GUI `session-…` checkpoint carries no such count, so rule 3 has no data
///   from it (a GUI log's rule-3 source is its value-fc1 lines);
/// - per-site counts come from what the lines name: `parkedBy=` (every
///   affected site) on logs that have the parked counts; otherwise only the
///   live line's `worst=` site between checkpoints (a lower bound) and
///   relu / leaky_relu sites only. Such a line's overall arm still divides
///   by every channel an activation consumes — the run's checkpoint table
///   (rows whose `act` is not `-`) — and has no data in a run without a
///   table: dividing by the relu / leaky_relu channels alone would overstate
///   the fraction on a mixed tower (B-silu: 20/144 instead of 20/1040);
/// - `[VS-UCI]` rows carry no `pIllM`, so rule 4 has no data offline;
/// - logs without step rows (GUI logs, whose `[STATS]` values are rolling
///   means) are replayed for the layer-health rules only.
///
/// **Runs.** Within one log, each `[RUN]` line after the log's first starts a
/// fresh evaluator, as the app starts a fresh monitor per run; the first
/// `[RUN]` of each later log passed together does not, so logs passed
/// together are one continuing evaluator across the log boundary (an offline
/// convenience: in the app each segment gets its own monitor). Channel counts
/// for live lines come from a pre-scan of every checkpoint table **per run**
/// (a run starts at each `[RUN]` line; lines before a log's first `[RUN]`
/// form a run of their own).
enum TrainingHealthLogReplay {

    struct Source: Sendable {
        let name: String
        let text: String
    }

    struct Options: Sendable {
        /// Use a row's `step=` as its trainer step when the log has no
        /// `trainerStep=` (explicit only; refused unless the rows' steps
        /// strictly increase).
        let segmentStepAsTrainerStep: Bool
        let config: TrainingHealthConfig
    }

    struct Output: Sendable {
        /// The limitations of the logs given, so a reader knows which rules
        /// could have fired.
        let header: [String]
        /// Every line the monitors wrote, in order.
        let lines: [TrainingHealthLogLine]
        /// Every committed event, in order.
        let events: [TrainingHealthEvent]
    }

    /// The first line of a bundled incident excerpt. Only in a file that
    /// starts with it are `#` lines skipped; a session log never has them.
    static let excerptHeaderMarker = "# training-health incident excerpt"

    // MARK: - Parsed lines

    struct StepRow: Sendable, Equatable {
        let segmentStep: Int
        let trainerStep: Int?
        let loss: Float
        let illegalMassPenalty: Float?
        let gradGlobalNorm: Float?
        let policyLogitMean: Float?
        let totalMs: Double?
        let learningRate: Double?
        let momentum: Double?
        /// Fields the row does not carry at all (as opposed to `--`).
        let absentFields: [String]
        let isTrainVsUci: Bool
    }

    /// The layer-health fields of a live line or checkpoint headline.
    struct CompactHealth: Sendable, Equatable {
        /// relu / leaky_relu classification (`reluSites`, `ch`, `dead`); nil
        /// counts when the log prints `n/a`.
        let reluClassifiedSiteCount: Int?
        let reluChannelCount: Int?
        let reluDeadCount: Int?
        let worstSite: String?
        let worstSiteDead: Int?
        /// Activation-aware parked fields; nil when the line predates them.
        let parked: Parked?
        let nonFiniteValueCount: Int?
        let runningVariance: LayerHealthDigest.RunningVarianceRunaway?
        let valueFC1: LayerHealthDigest.ValueFC1Velocity?

        struct Parked: Sendable, Equatable {
            /// 0 when the line says `parked=n/a`.
            let siteCount: Int
            let channelCount: Int
            let parkedCount: Int
            let sites: [(site: String, parked: Int, channels: Int)]

            static func == (lhs: Parked, rhs: Parked) -> Bool {
                lhs.siteCount == rhs.siteCount && lhs.channelCount == rhs.channelCount
                    && lhs.parkedCount == rhs.parkedCount
                    && lhs.sites.map(\.site) == rhs.sites.map(\.site)
                    && lhs.sites.map(\.parked) == rhs.sites.map(\.parked)
                    && lhs.sites.map(\.channels) == rhs.sites.map(\.channels)
            }
        }
    }

    struct TableRow: Sendable, Equatable {
        let site: String
        let activation: String
        let channelCount: Int
        /// nil when the table prints `n/a` (a site the relu classification
        /// does not cover).
        let deadCount: Int?
    }

    enum ParsedLine: Sendable {
        case run(build: String?)
        case app(build: String?)
        case stepRow(StepRow)
        case live(trainerStep: Int, health: CompactHealth)
        case liveFailed
        case checkpointHeadline(context: String, step: Int, trainerStep: Int?, health: CompactHealth)
        case checkpointFailed
        /// A `[LAYER-HEALTH]` detail line (indented), its text after the tag.
        case checkpointDetail(String)
        /// A dedicated value-FC1 read (D6), with the steps the writing
        /// process had trained when it read the state (`trained=`, rule 3's
        /// gate).
        case valueFC1(trainerStep: Int, stepsTrained: Int, zero: Int, units: Int)
        case other
    }

    // MARK: - Entry point

    static func run(_ sources: [Source], options: Options) throws -> Output {
        let parsed = try sources.map { try parse($0) }
        for file in parsed {
            let hasData = file.lines.contains { entry in
                switch entry.line {
                case .stepRow, .live, .liveFailed, .checkpointHeadline, .checkpointFailed, .valueFC1: return true
                case .run, .app, .checkpointDetail, .other: return false
                }
            }
            guard hasData else { throw TrainingHealthLogReplayError.noData(file: file.name) }
        }
        let prescan = try prescanRuns(parsed)
        var driver = Driver(options: options, prescan: prescan)
        let header = headerLines(sources: sources, parsed: parsed, options: options)
        for (sourceIndex, file) in parsed.enumerated() {
            try driver.replay(file, sourceIndex: sourceIndex)
        }
        driver.finish()
        return Output(header: header, lines: driver.lines, events: driver.events)
    }

    // MARK: - Parsing

    struct ParsedFile: Sendable {
        let name: String
        /// (1-based line number, parsed line)
        let lines: [(number: Int, line: ParsedLine)]
        let build: String?
    }

    static func parse(_ source: Source) throws -> ParsedFile {
        var rawLines = source.text.components(separatedBy: "\n")
        if rawLines.last == "" { rawLines.removeLast() }
        let isExcerpt = rawLines.first == excerptHeaderMarker
        var lines: [(number: Int, line: ParsedLine)] = []
        var build: String?
        for (index, raw) in rawLines.enumerated() {
            let number = index + 1
            if isExcerpt, raw.hasPrefix("#") { continue }
            let parsedLine = try parseLine(stripTimestamp(raw), file: source.name, number: number)
            switch parsedLine {
            case .run(let runBuild), .app(let runBuild):
                if build == nil { build = runBuild }
            default:
                break
            }
            lines.append((number, parsedLine))
        }
        return ParsedFile(name: source.name, lines: lines, build: build)
    }

    /// Drop the `HH:MM:SS.mmm  ` timestamp a session-log line starts with.
    static func stripTimestamp(_ line: String) -> Substring {
        let trimmed = line.hasSuffix("\r") ? line.dropLast() : Substring(line)
        guard let bracket = trimmed.firstIndex(of: "[") else { return trimmed }
        let prefix = trimmed[trimmed.startIndex..<bracket]
        guard prefix.allSatisfy({ $0.isNumber || $0 == ":" || $0 == "." || $0 == " " }) else { return trimmed }
        return trimmed[bracket...]
    }

    static func parseLine(_ content: Substring, file: String, number: Int) throws -> ParsedLine {
        func malformed(_ reason: String) -> TrainingHealthLogReplayError {
            .malformedLine(file: file, line: number, reason: reason)
        }
        if content.hasPrefix("[RUN] ") || content.hasPrefix("[APP] ") {
            // These lines carry prose and quoted argv; only `build=` is read.
            let build = content.split(separator: " ").first { $0.hasPrefix("build=") }.map { String($0.dropFirst("build=".count)) }
            return content.hasPrefix("[RUN] ") ? .run(build: build) : .app(build: build)
        }
        if content.hasPrefix("[REPLAY] step=") || content.hasPrefix("[VS-UCI] step=") {
            return .stepRow(try parseStepRow(content, malformed: malformed))
        }
        let tag = LayerHealthLog.tag
        guard content.hasPrefix(tag) else { return .other }
        let rest = content.dropFirst(tag.count)
        if rest.hasPrefix("   ") {
            return .checkpointDetail(String(rest))
        }
        if rest.hasPrefix(" live read failed") {
            return .liveFailed
        }
        if rest.hasPrefix(" live ") {
            let fields = try keyValueFields(rest.dropFirst(" live ".count), malformed: malformed)
            guard let stepText = fields["trainerStep"], let step = Int(stepText) else {
                throw malformed("live line without a trainerStep")
            }
            return .live(trainerStep: step, health: try compactHealth(fields, malformed: malformed))
        }
        if rest.hasPrefix(" checkpoint ") {
            let body = rest.dropFirst(" checkpoint ".count)
            if body.contains(" failed: ") { return .checkpointFailed }
            guard let space = body.firstIndex(of: " ") else { throw malformed("checkpoint headline without fields") }
            let context = String(body[body.startIndex..<space])
            let fields = try keyValueFields(body[space...], malformed: malformed)
            guard let stepText = fields["step"], let step = Int(stepText) else {
                throw malformed("checkpoint headline without step=")
            }
            let trainerStep = try fields["trainerStep"].map { text -> Int in
                guard let value = Int(text) else { throw malformed("trainerStep=\(text) is not an integer") }
                return value
            }
            return .checkpointHeadline(
                context: context, step: step, trainerStep: trainerStep,
                health: try compactHealth(fields, malformed: malformed))
        }
        if rest.hasPrefix(" value-fc1 ") {
            let fields = try keyValueFields(rest.dropFirst(" value-fc1 ".count), malformed: malformed)
            guard let stepText = fields["trainerStep"], let step = Int(stepText),
                  let trainedText = fields["trained"], let trained = Int(trainedText),
                  let velocity = try fraction(fields["valueFC1ZeroVel"], malformed: malformed) else {
                throw malformed("value-fc1 line without integer trainerStep=, trained= and valueFC1ZeroVel=")
            }
            return .valueFC1(trainerStep: step, stepsTrained: trained, zero: velocity.0, units: velocity.1)
        }
        return .other
    }

    /// Split `text` into `key=value` fields on spaces outside parentheses and
    /// brackets. A token without `=` is malformed unless it is a
    /// parenthesized annotation (`(no relu/leaky_relu BN sites)`). A
    /// repeated key is malformed.
    static func keyValueFields(
        _ text: Substring,
        malformed: (String) -> TrainingHealthLogReplayError
    ) throws -> [String: String] {
        var tokens: [String] = []
        var current = ""
        var depth = 0
        for character in text {
            switch character {
            case "(", "[":
                depth += 1
                current.append(character)
            case ")", "]":
                depth -= 1
                if depth < 0 { throw malformed("unbalanced \(character)") }
                current.append(character)
            case " " where depth == 0:
                if !current.isEmpty { tokens.append(current) }
                current = ""
            default:
                current.append(character)
            }
        }
        if depth != 0 { throw malformed("unbalanced parentheses") }
        if !current.isEmpty { tokens.append(current) }
        var fields: [String: String] = [:]
        for token in tokens {
            guard let equals = token.firstIndex(of: "=") else {
                if token.hasPrefix("(") && token.hasSuffix(")") { continue }
                throw malformed("token \"\(token)\" is not key=value")
            }
            let key = String(token[token.startIndex..<equals])
            let value = String(token[token.index(after: equals)...])
            if fields[key] != nil {
                throw malformed("field \(key) appears twice")
            }
            fields[key] = value
        }
        return fields
    }

    static func parseStepRow(_ content: Substring, malformed: (String) -> TrainingHealthLogReplayError) throws -> StepRow {
        let isTrainVsUci = content.hasPrefix("[VS-UCI] ")
        let fields = try keyValueFields(content.dropFirst("[REPLAY] ".count), malformed: malformed)
        guard let stepText = fields["step"], let step = Int(stepText) else {
            throw malformed("step row without an integer step=")
        }
        var absent: [String] = []
        func number<T: LosslessStringConvertible>(_ key: String, as type: T.Type) throws -> T? {
            guard let text = fields[key] else {
                absent.append(key)
                return nil
            }
            if text == TrainingHealthLog.notMeasured { return nil }
            guard let value = T(text) else { throw malformed("\(key)=\(text) is not a number") }
            return value
        }
        guard let lossText = fields["loss"] else { throw malformed("step row without loss=") }
        guard let loss = Float(lossText) else { throw malformed("loss=\(lossText) is not a number") }
        let trainerStep = try number("trainerStep", as: Int.self)
        let illegal = try number("pIllM", as: Float.self)
        let gradient = try number("gNorm", as: Float.self)
        let offset = try number("pLogitMean", as: Float.self)
        let ms = try number("ms", as: Double.self)
        let lr = try number("lr", as: Double.self)
        let mom = try number("mom", as: Double.self)
        return StepRow(
            segmentStep: step, trainerStep: trainerStep, loss: loss, illegalMassPenalty: illegal,
            gradGlobalNorm: gradient, policyLogitMean: offset, totalMs: ms, learningRate: lr,
            momentum: mom, absentFields: absent, isTrainVsUci: isTrainVsUci)
    }

    /// `a/b` as two integers; nil for nil.
    static func fraction(
        _ text: String?,
        malformed: (String) -> TrainingHealthLogReplayError
    ) throws -> (Int, Int)? {
        guard let text else { return nil }
        let parts = text.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2, let a = Int(parts[0]), let b = Int(parts[1]) else {
            throw malformed("\"\(text)\" is not a/b")
        }
        return (a, b)
    }

    static func compactHealth(
        _ fields: [String: String],
        malformed: (String) -> TrainingHealthLogReplayError
    ) throws -> CompactHealth {
        func integer(_ key: String) throws -> Int? {
            guard let text = fields[key], text != "n/a" else { return nil }
            guard let value = Int(text) else { throw malformed("\(key)=\(text) is not an integer") }
            return value
        }
        let reluSites = try fraction(fields["reluSites"], malformed: malformed)
        let reluDead = try integer("dead")
        let reluChannels = try integer("ch")

        var worstSite: String?
        var worstDead: Int?
        if let worst = fields["worst"], worst != "none" {
            // `site(dead d off o on a)`
            guard let open = worst.firstIndex(of: "("), worst.hasSuffix(")") else {
                throw malformed("worst=\(worst) is not site(dead d off o on a)")
            }
            worstSite = String(worst[worst.startIndex..<open])
            let inner = worst[worst.index(after: open)..<worst.index(before: worst.endIndex)].split(separator: " ")
            guard inner.count == 6, inner[0] == "dead", let dead = Int(inner[1]) else {
                throw malformed("worst=\(worst) is not site(dead d off o on a)")
            }
            worstDead = dead
        }

        var parked: CompactHealth.Parked?
        if fields["parked"] == "n/a" {
            parked = CompactHealth.Parked(siteCount: 0, channelCount: 0, parkedCount: 0, sites: [])
        } else if let parkedText = fields["parked"] {
            guard let count = Int(parkedText), let sitesField = try fraction(fields["parkedSites"], malformed: malformed),
                  let channels = try integer("parkedCh"), let by = fields["parkedBy"] else {
                throw malformed("parked fields incomplete (parked, parkedSites, parkedCh, parkedBy)")
            }
            var sites: [(site: String, parked: Int, channels: Int)] = []
            if by != "none" {
                for entry in by.split(separator: ",") {
                    guard let colon = entry.lastIndex(of: ":"),
                          let counts = try fraction(String(entry[entry.index(after: colon)...]), malformed: malformed) else {
                        throw malformed("parkedBy entry \"\(entry)\" is not site:parked/channels")
                    }
                    sites.append((String(entry[entry.startIndex..<colon]), counts.0, counts.1))
                }
            }
            parked = CompactHealth.Parked(siteCount: sitesField.0, channelCount: channels, parkedCount: count, sites: sites)
        }

        var runningVariance: LayerHealthDigest.RunningVarianceRunaway?
        if let text = fields["rvMaxOverMedian"], text != "n/a" {
            guard let at = text.firstIndex(of: "@"), let ratio = Double(text[text.startIndex..<at]) else {
                throw malformed("rvMaxOverMedian=\(text) is not value@site[channel]")
            }
            let siteAndChannel = text[text.index(after: at)...]
            let site = siteAndChannel.firstIndex(of: "[").map { String(siteAndChannel[siteAndChannel.startIndex..<$0]) }
                ?? String(siteAndChannel)
            runningVariance = LayerHealthDigest.RunningVarianceRunaway(maxOverMedian: ratio, site: site)
        }

        let velocity = try fraction(fields["valueFC1ZeroVel"], malformed: malformed)
        return CompactHealth(
            reluClassifiedSiteCount: reluSites?.0,
            reluChannelCount: reluChannels,
            reluDeadCount: reluDead,
            worstSite: worstSite,
            worstSiteDead: worstDead,
            parked: parked,
            nonFiniteValueCount: try integer("nonFinite"),
            runningVariance: runningVariance,
            valueFC1: velocity.map { LayerHealthDigest.ValueFC1Velocity(zeroVelocityUnitCount: $0.0, unitCount: $0.1) })
    }

    /// A batch-norm table row: `site act ch dead off on zeroγ nonfin …`.
    static func tableRow(_ detail: String) -> TableRow? {
        let tokens = detail.split(separator: " ", omittingEmptySubsequences: true)
        guard tokens.count >= 4, let channels = Int(tokens[2]) else { return nil }
        let dead: Int?
        if tokens[3] == "n/a" {
            dead = nil
        } else if let value = Int(tokens[3]) {
            dead = value
        } else {
            return nil
        }
        return TableRow(site: String(tokens[0]), activation: String(tokens[1]), channelCount: channels, deadCount: dead)
    }

    // MARK: - Pre-scan (per run)

    struct RunKey: Hashable, Sendable {
        let source: Int
        let ordinal: Int
    }

    /// One batch-norm table row as the pre-scan keeps it: the site's channel
    /// count and the activation that consumes it (`-` for none), with where
    /// it was first read.
    struct TableSite: Sendable {
        let channelCount: Int
        let activation: String
        let file: String
        let line: Int
    }

    struct RunFacts: Sendable {
        var tableSites: [String: TableSite] = [:]
        var valueFC1Activation: String?
        var hasValueFC1Lines = false

        /// Sites an activation consumes (table rows whose `act` is not `-`);
        /// nil when the run has no checkpoint table.
        var modeledSiteCount: Int? {
            tableSites.isEmpty ? nil : tableSites.values.filter { $0.activation != Self.noActivation }.count
        }

        /// Their channels: rule 2's overall denominator for a line that
        /// predates the parked counts; nil when the run has no table.
        var modeledChannelCount: Int? {
            tableSites.isEmpty
                ? nil
                : tableSites.values.filter { $0.activation != Self.noActivation }.reduce(0) { $0 + $1.channelCount }
        }

        /// How a table row says no activation consumes the site.
        static let noActivation = "-"
    }

    /// Every checkpoint table's sites (channel count and consuming
    /// activation), and the value FC1 layer's activation, per run. Two
    /// different channel counts or activations for one site within one run
    /// are a malformed input.
    static func prescanRuns(_ files: [ParsedFile]) throws -> [RunKey: RunFacts] {
        var facts: [RunKey: RunFacts] = [:]
        for (sourceIndex, file) in files.enumerated() {
            var ordinal = 0
            var inTable = false
            var inVelocity = false
            for (number, line) in file.lines {
                let key = RunKey(source: sourceIndex, ordinal: ordinal)
                switch line {
                case .run:
                    ordinal += 1
                    inTable = false
                    inVelocity = false
                case .checkpointDetail(let detail):
                    let trimmed = detail.drop(while: { $0 == " " })
                    let isSectionLine = !detail.hasPrefix("     ")
                    if isSectionLine {
                        inTable = trimmed.hasPrefix("batch-norm sites")
                        inVelocity = trimmed.hasPrefix("FC hidden-unit velocity")
                        continue
                    }
                    if inTable {
                        if trimmed.hasPrefix("site ") { continue }
                        guard let row = tableRow(String(trimmed)) else {
                            throw TrainingHealthLogReplayError.malformedLine(
                                file: file.name, line: number, reason: "batch-norm table row does not parse")
                        }
                        var runFacts = facts[key, default: RunFacts()]
                        if let existing = runFacts.tableSites[row.site] {
                            if existing.channelCount != row.channelCount {
                                throw TrainingHealthLogReplayError.conflictingChannelCounts(
                                    site: row.site, first: "\(existing.file):\(existing.line) (\(existing.channelCount))",
                                    second: "\(file.name):\(number) (\(row.channelCount))")
                            }
                            if existing.activation != row.activation {
                                throw TrainingHealthLogReplayError.conflictingActivations(
                                    site: row.site, first: "\(existing.file):\(existing.line) (\(existing.activation))",
                                    second: "\(file.name):\(number) (\(row.activation))")
                            }
                        } else {
                            runFacts.tableSites[row.site] = TableSite(
                                channelCount: row.channelCount, activation: row.activation, file: file.name, line: number)
                        }
                        facts[key] = runFacts
                    } else if inVelocity, trimmed.hasPrefix("value.fc1.weight ") {
                        let tokens = trimmed.split(separator: " ", omittingEmptySubsequences: true)
                        guard tokens.count >= 2 else {
                            throw TrainingHealthLogReplayError.malformedLine(
                                file: file.name, line: number, reason: "value.fc1.weight velocity row without an activation")
                        }
                        facts[key, default: RunFacts()].valueFC1Activation = String(tokens[1])
                    }
                case .valueFC1:
                    facts[key, default: RunFacts()].hasValueFC1Lines = true
                default:
                    inTable = false
                    inVelocity = false
                }
            }
        }
        return facts
    }

    // MARK: - Header

    static func headerLines(sources: [Source], parsed: [ParsedFile], options: Options) -> [String] {
        var header: [String] = []
        header.append("offline replay of \(sources.count) log(s): \(sources.map(\.name).joined(separator: ", "))")
        header.append("logs passed together are one continuing evaluator across log boundaries; each [RUN] after a log's first starts a fresh evaluator")
        header.append("sparse semantics: each step row is one live evaluation with a one-record window; spike references are the rows in the 1000 trainer steps before it")
        if options.segmentStepAsTrainerStep {
            header.append("--segment-step-as-trainer-step: rows without trainerStep= use step=; trainer steps may be offset by the start model's clock (spans exact, the learning gate unreliable)")
        }
        let ruleInputs: [(field: String, rule: TrainingHealthRule)] = [
            ("pIllM", .illegalMass), ("gNorm", .gradientCollapse), ("gNorm", .gradientSpike),
            ("pLogitMean", .policyOffsetDrift),
        ]
        for file in parsed {
            var rows = 0
            var vsUciRows = 0
            var absentCounts: [String: Int] = [:]
            var liveLines = 0
            var liveWithoutParked = 0
            var valueFC1Lines = 0
            for (_, line) in file.lines {
                switch line {
                case .stepRow(let row):
                    rows += 1
                    if row.isTrainVsUci { vsUciRows += 1 }
                    for field in row.absentFields { absentCounts[field, default: 0] += 1 }
                case .live(_, let health):
                    liveLines += 1
                    if health.parked == nil { liveWithoutParked += 1 }
                case .valueFC1:
                    valueFC1Lines += 1
                default:
                    break
                }
            }
            if rows == 0 {
                // Rule 3 needs the steps trained by the writing process: a
                // GUI `session-…` checkpoint has none, so only the dedicated
                // value-fc1 lines (`trained=`) and CLI checkpoints feed it.
                header.append("\(file.name): no [REPLAY]/[VS-UCI] step rows: layer-health rules only (non_finite, dead_channels, bn_running_variance_runaway); value_fc1_zero_velocity only from [LAYER-HEALTH] value-fc1 lines and replay-/vsuci- checkpoints (\(valueFC1Lines) value-fc1 lines in this log)")
            }
            for input in ruleInputs {
                guard let count = absentCounts[input.field], count > 0 else { continue }
                header.append("\(file.name): \(input.field) absent in \(count) of \(rows) rows: \(input.rule.rawValue) no data on them")
            }
            if vsUciRows > 0 {
                header.append("\(file.name): [VS-UCI] rows carry no pIllM: illegal_mass no data offline")
            }
            if liveWithoutParked > 0 {
                header.append("\(file.name): \(liveWithoutParked) of \(liveLines) live lines predate the parked counts: dead_channels counts relu/leaky_relu sites only there (a lower bound), per-site counts between checkpoints are the worst= site only (a lower bound), and the overall arm divides by every activated channel of the run's checkpoint table (no data in a run without one)")
            }
        }
        return header
    }

    // MARK: - Driver

    struct Driver {
        let options: Options
        let prescan: [RunKey: RunFacts]
        private(set) var lines: [TrainingHealthLogLine] = []
        private(set) var events: [TrainingHealthEvent] = []

        /// Created at first use, with the current run's value-FC1
        /// applicability; nil after a `[RUN]` that starts a fresh evaluator.
        private var currentMonitor: TrainingHealthMonitor?
        private let sink: LineCollector
        private var runKey = RunKey(source: 0, ordinal: 0)
        /// Steps-trained offset of the logs already replayed into this
        /// monitor (their last segment step).
        private var stepsTrainedOffset = 0
        private var lastSegmentStepInSource = 0
        private var lastRowStepsTrained = 0
        private var lastObservedTrainerStep: Int?
        private var pendingRow: (row: StepRow, trainerStep: Int)?
        private var pendingCheckpoint: PendingCheckpoint?
        private var lastRowSegmentStep: Int?

        struct PendingCheckpoint {
            let context: String
            let step: Int
            let trainerStep: Int
            let health: CompactHealth
            var tableRows: [TableRow] = []
            var inTable = false
        }

        /// Collects the monitor's lines (the sink is called under the
        /// monitor's lock, synchronously, from this thread).
        final class LineCollector: @unchecked Sendable {
            private let box = SyncBox<[TrainingHealthLogLine]>([])
            func append(_ line: TrainingHealthLogLine) { box.modify { $0.append(line) } }
            func drain() -> [TrainingHealthLogLine] { box.mutate { lines in defer { lines.removeAll() }; return lines } }
        }

        init(options: Options, prescan: [RunKey: RunFacts]) {
            self.options = options
            self.prescan = prescan
            self.sink = LineCollector()
        }

        /// The current evaluator's monitor, created (and its `[HEALTH]
        /// config` line logged) on first use.
        private mutating func monitor() -> TrainingHealthMonitor {
            if let currentMonitor { return currentMonitor }
            let created = TrainingHealthMonitor(valueFC1Applicability: Self.applicability(prescan[runKey]))
            currentMonitor = created
            lines.append(TrainingHealthLogLine(
                text: TrainingHealthLog.configLine(
                    config: options.config, path: "offline", valueFC1Applicability: created.valueFC1Applicability),
                eventKind: nil))
            return created
        }

        static func applicability(_ facts: RunFacts?) -> TrainingHealthValueFC1Applicability {
            if let name = facts?.valueFC1Activation {
                guard let activation = ActivationFunction(rawValue: name) else { return .unknownFromLog }
                return TrainingHealthValueFC1Applicability(valueFC1Activation: activation)
            }
            // The dedicated value-FC1 read (D6) is made only for a relu layer.
            return facts?.hasValueFC1Lines == true ? .applies : .unknownFromLog
        }

        private var logSink: TrainingHealthLogSink {
            let collector = sink
            return { collector.append($0) }
        }

        private mutating func collect(_ result: TrainingHealthEvaluation?) {
            lines.append(contentsOf: sink.drain())
            if let result { events.append(contentsOf: result.events) }
        }

        mutating func replay(_ file: ParsedFile, sourceIndex: Int) throws {
            var ordinal = 0
            var sawRunInThisSource = false
            if sourceIndex > 0 {
                stepsTrainedOffset += lastSegmentStepInSource
                lastSegmentStepInSource = 0
                lastRowSegmentStep = nil
            }
            runKey = RunKey(source: sourceIndex, ordinal: 0)
            for (number, line) in file.lines {
                if case .checkpointDetail(let detail) = line, pendingCheckpoint != nil {
                    absorbDetail(detail)
                    continue
                }
                try finishCheckpoint()
                switch line {
                case .run:
                    flushRow()
                    ordinal += 1
                    runKey = RunKey(source: sourceIndex, ordinal: ordinal)
                    let continuesAcrossLogs = sourceIndex > 0 && !sawRunInThisSource
                    sawRunInThisSource = true
                    if !continuesAcrossLogs {
                        startFreshMonitor()
                    }
                case .app:
                    break
                case .stepRow(let row):
                    try record(row, file: file, number: number)
                case .live(let trainerStep, let health):
                    evaluateLive(trainerStep: trainerStep, health: health)
                case .liveFailed:
                    if let pending = pendingRow {
                        pendingRow = nil
                        evaluatePendingRow(pending, layerHealth: .readFailed)
                    }
                case .checkpointHeadline(let context, let step, let trainerStep, let health):
                    flushRow()
                    guard let clock = trainerStep ?? (options.segmentStepAsTrainerStep ? step : nil) else {
                        throw TrainingHealthLogReplayError.unsupportedFormat(
                            file: file.name, build: file.build, missingField: "trainerStep (checkpoint headline)")
                    }
                    lastSegmentStepInSource = max(lastSegmentStepInSource, contextIsCLI(context) ? step : 0)
                    pendingCheckpoint = PendingCheckpoint(context: context, step: step, trainerStep: clock, health: health)
                case .checkpointFailed:
                    flushRow()
                case .checkpointDetail:
                    break
                case .valueFC1(let trainerStep, let stepsTrained, let zero, let units):
                    flushRow()
                    lastSegmentStepInSource = max(lastSegmentStepInSource, stepsTrained)
                    evaluateValueFC1(trainerStep: trainerStep, stepsTrained: stepsTrained, zero: zero, units: units)
                case .other:
                    break
                }
            }
            try finishCheckpoint()
            flushRow()
        }

        mutating func finish() {
            flushRow()
            currentMonitor?.writeFinalCheck(log: logSink)
            collect(nil)
        }

        private mutating func startFreshMonitor() {
            flushRow()
            currentMonitor?.writeFinalCheck(log: logSink)
            collect(nil)
            currentMonitor = nil
            stepsTrainedOffset = 0
            lastSegmentStepInSource = 0
            lastRowStepsTrained = 0
            lastObservedTrainerStep = nil
            lastRowSegmentStep = nil
        }

        private func contextIsCLI(_ context: String) -> Bool {
            context.hasPrefix("replay-") || context.hasPrefix("vsuci-")
        }

        private mutating func stamp(stepsTrained: Int) -> TrainingHealthStamp {
            let base = monitor().observationStamp()
            return TrainingHealthStamp(
                runID: base.runID, generation: base.generation,
                stepsTrainedByThisProcess: stepsTrained, lastRecordedTrainerStep: base.lastRecordedTrainerStep)
        }

        private mutating func announceRewindIfNeeded(trainerStep: Int, restoredClock: Int) {
            if let last = lastObservedTrainerStep, trainerStep <= last {
                monitor().noteTrainerClockRewind(to: restoredClock, log: logSink)
                collect(nil)
            }
            lastObservedTrainerStep = trainerStep
        }

        private mutating func record(_ row: StepRow, file: ParsedFile, number: Int) throws {
            flushRow()
            let trainerStep: Int
            if let recorded = row.trainerStep {
                trainerStep = recorded
            } else if options.segmentStepAsTrainerStep {
                if let previous = lastRowSegmentStep, row.segmentStep <= previous {
                    throw TrainingHealthLogReplayError.nonIncreasingSegmentSteps(file: file.name, line: number)
                }
                trainerStep = row.segmentStep
            } else {
                throw TrainingHealthLogReplayError.unsupportedFormat(
                    file: file.name, build: file.build, missingField: "trainerStep")
            }
            lastRowSegmentStep = row.segmentStep
            lastSegmentStepInSource = max(lastSegmentStepInSource, row.segmentStep)
            announceRewindIfNeeded(trainerStep: trainerStep, restoredClock: trainerStep - 1)
            monitor().recordStep(TrainingHealthStepRecord(
                trainerStep: trainerStep, loss: row.loss, illegalMassPenalty: row.illegalMassPenalty,
                gradGlobalNorm: row.gradGlobalNorm, totalMs: row.totalMs, policyLogitMean: row.policyLogitMean))
            lastRowStepsTrained = stepsTrainedOffset + row.segmentStep
            pendingRow = (row, trainerStep)
        }

        private mutating func flushRow() {
            guard let pending = pendingRow else { return }
            pendingRow = nil
            evaluatePendingRow(pending, layerHealth: .absentFromLog)
        }

        private mutating func evaluatePendingRow(
            _ pending: (row: StepRow, trainerStep: Int),
            layerHealth: TrainingHealthLiveLayerHealth
        ) {
            let observationStamp = stamp(stepsTrained: stepsTrainedOffset + pending.row.segmentStep)
            let result = monitor().evaluateLive(
                stamp: observationStamp,
                layerHealth: layerHealth, learningRate: pending.row.learningRate,
                momentum: pending.row.momentum, config: options.config, log: logSink)
            collect(result)
        }

        private mutating func evaluateLive(trainerStep: Int, health: CompactHealth) {
            let digest = Self.digest(health, tier: .live, tableRows: nil, runFacts: prescan[runKey])
            if let pending = pendingRow, pending.trainerStep == trainerStep {
                pendingRow = nil
                evaluatePendingRow(pending, layerHealth: .read(digest, trainerStep: trainerStep))
                return
            }
            flushRow()
            announceRewindIfNeeded(trainerStep: trainerStep, restoredClock: trainerStep)
            let observationStamp = stamp(stepsTrained: lastRowStepsTrained)
            let result = monitor().evaluateLive(
                stamp: observationStamp, layerHealth: .read(digest, trainerStep: trainerStep),
                learningRate: nil, momentum: nil, config: options.config, log: logSink)
            collect(result)
        }

        private mutating func absorbDetail(_ detail: String) {
            guard var pending = pendingCheckpoint else { return }
            let trimmed = detail.drop(while: { $0 == " " })
            if !detail.hasPrefix("     ") {
                pending.inTable = trimmed.hasPrefix("batch-norm sites")
            } else if pending.inTable, !trimmed.hasPrefix("site "), let row = tableRow(String(trimmed)) {
                pending.tableRows.append(row)
            }
            pendingCheckpoint = pending
        }

        private mutating func finishCheckpoint() throws {
            guard let pending = pendingCheckpoint else { return }
            pendingCheckpoint = nil
            var digest = Self.digest(
                pending.health, tier: .checkpoint, tableRows: pending.tableRows, runFacts: prescan[runKey])
            let isCLI = contextIsCLI(pending.context)
            if !isCLI {
                digest = LayerHealthDigest(
                    tier: digest.tier, deadChannels: digest.deadChannels,
                    nonFiniteValueCount: digest.nonFiniteValueCount, runningVariance: digest.runningVariance,
                    valueFC1: nil)
            }
            let observationStamp = stamp(stepsTrained: isCLI ? stepsTrainedOffset + pending.step : 0)
            let result = monitor().evaluateCheckpoint(
                stamp: observationStamp,
                layerHealth: digest, digestTrainerStep: pending.trainerStep,
                config: options.config, log: logSink)
            collect(result)
        }

        /// `stepsTrained` is the writing process's own count (`trained=`);
        /// like a step row's segment step it is offset by the logs already
        /// replayed into this evaluator.
        private mutating func evaluateValueFC1(trainerStep: Int, stepsTrained: Int, zero: Int, units: Int) {
            let observationStamp = stamp(stepsTrained: stepsTrainedOffset + stepsTrained)
            let result = monitor().evaluateCheckpoint(
                stamp: observationStamp,
                layerHealth: .valueFC1Only(LayerHealthDigest.ValueFC1Velocity(zeroVelocityUnitCount: zero, unitCount: units)),
                digestTrainerStep: trainerStep, config: options.config, log: logSink)
            collect(result)
        }

        /// The digest of one parsed line: parked counts when the line has
        /// them (every activation, every affected site named); otherwise the
        /// relu / leaky_relu dead counts, per site from the checkpoint table
        /// or, for a live line, the worst site with its channel count from
        /// the run's pre-scan, over the run's modeled channel count (nil
        /// without a table: the overall arm then has no data).
        static func digest(
            _ health: CompactHealth,
            tier: LayerHealthDigest.Tier,
            tableRows: [TableRow]?,
            runFacts: RunFacts?
        ) -> LayerHealthDigest {
            let dead: LayerHealthDigest.DeadChannels?
            if let parked = health.parked {
                dead = LayerHealthDigest.DeadChannels(
                    modeledSiteCount: parked.siteCount,
                    modeledChannelCount: parked.channelCount,
                    parkedChannelCount: parked.parkedCount,
                    sites: parked.sites.map {
                        LayerHealthDigest.SiteDeadChannels(site: $0.site, parkedChannelCount: $0.parked, channelCount: $0.channels)
                    },
                    coversEveryActivation: true)
            } else if let siteCount = health.reluClassifiedSiteCount {
                var sites: [LayerHealthDigest.SiteDeadChannels] = []
                if let tableRows, !tableRows.isEmpty {
                    sites = tableRows.compactMap { row in
                        guard let dead = row.deadCount else { return nil }
                        return LayerHealthDigest.SiteDeadChannels(site: row.site, parkedChannelCount: dead, channelCount: row.channelCount)
                    }
                } else if let worst = health.worstSite, let worstDead = health.worstSiteDead {
                    sites = [LayerHealthDigest.SiteDeadChannels(
                        site: worst, parkedChannelCount: worstDead, channelCount: runFacts?.tableSites[worst]?.channelCount)]
                }
                if siteCount == 0 {
                    // `reluSites=0/n`: no relu / leaky_relu site, and the line
                    // predates the parked counts, so silu / gelu sites were
                    // never checked — no data, not "does not apply".
                    dead = nil
                } else if health.reluChannelCount != nil, let deadCount = health.reluDeadCount {
                    // The line's `ch=` counts relu / leaky_relu channels only;
                    // it is never the overall denominator.
                    dead = LayerHealthDigest.DeadChannels(
                        modeledSiteCount: runFacts?.modeledSiteCount,
                        modeledChannelCount: runFacts?.modeledChannelCount,
                        parkedChannelCount: deadCount, sites: sites, coversEveryActivation: false)
                } else {
                    dead = nil
                }
            } else {
                dead = nil
            }
            return LayerHealthDigest(
                tier: tier, deadChannels: dead, nonFiniteValueCount: health.nonFiniteValueCount,
                runningVariance: health.runningVariance, valueFC1: tier == .checkpoint ? health.valueFC1 : nil)
        }
    }
}

enum TrainingHealthLogReplayError: LocalizedError, Equatable {
    case malformedLine(file: String, line: Int, reason: String)
    case unsupportedFormat(file: String, build: String?, missingField: String)
    case conflictingChannelCounts(site: String, first: String, second: String)
    case conflictingActivations(site: String, first: String, second: String)
    case nonIncreasingSegmentSteps(file: String, line: Int)
    case noData(file: String)

    var errorDescription: String? {
        switch self {
        case .malformedLine(let file, let line, let reason):
            return "malformed line: \(file):\(line): \(reason)"
        case .unsupportedFormat(let file, let build, let field):
            return "unsupported log format: \(file) (build \(build ?? "unknown") from its [APP]/[RUN] line): step rows carry no \(field)"
        case .conflictingChannelCounts(let site, let first, let second):
            return "malformed input: site \(site) has two channel counts within one run: \(first) and \(second)"
        case .conflictingActivations(let site, let first, let second):
            return "malformed input: site \(site) has two activations within one run: \(first) and \(second)"
        case .nonIncreasingSegmentSteps(let file, let line):
            return "--segment-step-as-trainer-step refused: \(file):\(line): step= does not strictly increase"
        case .noData(let file):
            return "\(file): neither step rows nor [LAYER-HEALTH] lines"
        }
    }
}
