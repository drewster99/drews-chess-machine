import SwiftUI

/// Rich Load Session picker (replaces the bare `.fileImporter` as the
/// File > Load Session… front door; Browse… keeps the importer reachable
/// for sessions outside the default directory).
///
/// Layout: lineage-grouped list on the left (one section per run, newest
/// run first), detail pane for the selected save on the right, footer
/// with Cancel / Browse… / Load. Digits are monospaced and columns
/// aligned per the project UI rules; unreadable sessions render as
/// error rows rather than disappearing.
struct SessionPickerSheet: View {
    @Bindable var model: SessionPickerModel
    let onLoad: (SessionManifest) -> Void
    let onBrowse: () -> Void
    let onCancel: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            HStack(spacing: 0) {
                list
                    .frame(minWidth: 560)
                Divider()
                detail
                    .frame(width: 360)
            }
            Divider()
            footer
        }
        .frame(minWidth: 980, minHeight: 560)
    }

    // MARK: Header

    private var header: some View {
        HStack {
            Text("Load Session")
                .font(.title3.weight(.semibold))
            Spacer()
            if model.isScanning {
                ProgressView()
                    .controlSize(.small)
                Text("Indexing \(model.scannedCount)/\(model.totalCount)…")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            } else {
                Text("\(model.manifests.count) saves · \(model.groups.count) runs")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    // MARK: List

    private var list: some View {
        List(selection: $model.selectedID) {
            ForEach(model.groups) { group in
                Section {
                    ForEach(group.sessions) { m in
                        SessionPickerRow(manifest: m)
                            .tag(m.id)
                    }
                } header: {
                    groupHeader(group)
                }
            }
        }
        .listStyle(.inset)
    }

    @ViewBuilder
    private func groupHeader(_ group: SessionPickerModel.RunGroup) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            HStack(spacing: 8) {
                Text(group.lineageTag)
                    .font(.system(.callout, design: .monospaced).weight(.bold))
                Text("\(group.sessions.count) save\(group.sessions.count == 1 ? "" : "s")")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            if let arch = group.architectureSummary {
                Text(arch)
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
        .padding(.vertical, 2)
    }

    // MARK: Detail

    @ViewBuilder
    private var detail: some View {
        if let m = model.selectedManifest {
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    Text(m.folderName)
                        .font(.system(.caption, design: .monospaced).weight(.semibold))
                        .textSelection(.enabled)
                    if let err = m.loadError {
                        Label(err, systemImage: "exclamationmark.triangle.fill")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }
                    detailSection("Architecture", [
                        ("Summary", m.architectureSummary),
                        ("Parameters", m.parameterCount.map { $0.formatted() }),
                        ("Blocks × channels", zipDescribe(m.numBlocks, m.channels)),
                        ("Input planes", m.inputPlanes.map(String.init))
                    ])
                    detailSection("Run", [
                        ("Champion", m.championID),
                        ("Trainer", m.trainerID),
                        ("Steps", m.trainingSteps.map { $0.formatted() }),
                        ("Active training", m.elapsedTrainingSec.map(Self.formatHMS)),
                        ("Games emitted", m.emittedGames.map { $0.formatted() }),
                        ("Buffer", bufferDescribe(m)),
                        ("Build", buildDescribe(m))
                    ])
                    detailSection("Performance", [
                        ("Arenas / promotions", zipDescribe(m.arenaCount, m.promotionCount, sep: " / ")),
                        ("pElo (200 set)", m.latestPElo200.map { String(format: "%.0f", $0) }),
                        ("pElo (wide set)", m.latestPEloWide.map { String(format: "%.0f", $0) }),
                        ("Checkmates W/B", zipDescribe(m.whiteCheckmates, m.blackCheckmates, sep: " / ")),
                        ("Draws", m.drawCount.map { $0.formatted() })
                    ])
                    detailSection("Hyperparameters at save", [
                        ("Learning rate", m.learningRate.map { String(format: "%.2e", $0) }),
                        ("Batch size", m.batchSize.map(String.init)),
                        ("Weight decay", m.weightDecay.map { String(format: "%.1e", $0) }),
                        ("Dropout rate", m.dropoutRate.map { String(format: "%.2f", $0) }),
                        ("Momentum", m.momentumCoeff.map { String(format: "%.2f", $0) }),
                        ("Promote ≥", m.promoteThreshold.map { String(format: "%.2f", $0) }),
                        ("Self-play workers", m.selfPlayWorkerCount.map(String.init))
                    ])
                    if let url = model.folderURL(for: m) {
                        Button {
                            NSWorkspace.shared.activateFileViewerSelecting([url])
                        } label: {
                            Label("Reveal in Finder", systemImage: "folder")
                        }
                        .buttonStyle(.link)
                    }
                }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        } else {
            VStack {
                Spacer()
                Text("Select a save")
                    .foregroundStyle(.secondary)
                Spacer()
            }
            .frame(maxWidth: .infinity)
        }
    }

    @ViewBuilder
    private func detailSection(_ title: String, _ rows: [(String, String?)]) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title.uppercased())
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
            ForEach(rows, id: \.0) { row in
                HStack(alignment: .firstTextBaseline) {
                    Text(row.0)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(width: 130, alignment: .leading)
                    Text(row.1 ?? "—")
                        .font(.system(.caption, design: .monospaced))
                        .monospacedDigit()
                        .textSelection(.enabled)
                }
            }
        }
    }

    private func zipDescribe(_ a: Int?, _ b: Int?, sep: String = " × ") -> String? {
        switch (a, b) {
        case (nil, nil): return nil
        default: return "\(a.map(String.init) ?? "—")\(sep)\(b.map(String.init) ?? "—")"
        }
    }

    private func bufferDescribe(_ m: SessionManifest) -> String? {
        guard let stored = m.bufferStored else { return nil }
        if let cap = m.bufferCapacity, cap > 0 {
            return "\(stored.formatted()) / \(cap.formatted())"
        }
        return stored.formatted()
    }

    private func buildDescribe(_ m: SessionManifest) -> String? {
        guard let n = m.buildNumber else { return nil }
        let git = m.buildGitHash.map { h in
            " (\(h)\(m.buildGitDirty == true ? "*" : ""))"
        } ?? ""
        return "\(n)\(git)"
    }

    // MARK: Footer

    private var footer: some View {
        HStack {
            Button("Browse…", action: onBrowse)
            Spacer()
            Button("Cancel", role: .cancel, action: onCancel)
            Button("Load") {
                if let m = model.selectedManifest { onLoad(m) }
            }
            .keyboardShortcut(.defaultAction)
            .disabled(model.selectedManifest == nil || model.selectedManifest?.loadError != nil)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    static func formatHMS(_ seconds: Double) -> String {
        let s = Int(seconds.rounded())
        return String(format: "%d:%02d:%02d", s / 3600, (s % 3600) / 60, s % 60)
    }
}

/// One save in the picker list: aligned columns of the at-a-glance facts.
struct SessionPickerRow: View {
    let manifest: SessionManifest

    private static let dateFmt: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "MMM d, h:mm a"
        return f
    }()

    var body: some View {
        HStack(spacing: 10) {
            Text(manifest.savedAt.map { Self.dateFmt.string(from: $0) } ?? "—")
                .frame(width: 110, alignment: .leading)
            triggerBadge
                .frame(width: 80, alignment: .leading)
            Text(manifest.trainingSteps.map { $0.formatted() } ?? "—")
                .frame(width: 70, alignment: .trailing)
            Text(arenaSummary)
                .frame(width: 60, alignment: .trailing)
            Text(manifest.latestPEloWide.map { String(format: "%.0f", $0) } ?? "—")
                .frame(width: 50, alignment: .trailing)
            Text(manifest.buildNumber.map(String.init) ?? "—")
                .frame(width: 46, alignment: .trailing)
            Text(diskString)
                .frame(width: 60, alignment: .trailing)
            if manifest.loadError != nil {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
            }
            Spacer(minLength: 0)
        }
        .font(.system(.caption, design: .monospaced))
        .monospacedDigit()
        .lineLimit(1)
    }

    private var arenaSummary: String {
        let a = manifest.arenaCount.map(String.init) ?? "—"
        let p = manifest.promotionCount.map(String.init) ?? "—"
        return "\(a)/\(p)"
    }

    private var diskString: String {
        guard let b = manifest.diskBytes else { return "—" }
        return String(format: "%.1f GB", Double(b) / 1_073_741_824.0)
    }

    @ViewBuilder
    private var triggerBadge: some View {
        let t = manifest.trigger ?? "?"
        Text(t)
            .font(.system(.caption2, design: .monospaced).weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 1)
            .background(
                RoundedRectangle(cornerRadius: 4)
                    .fill(badgeColor(t).opacity(0.18))
            )
            .foregroundStyle(badgeColor(t))
    }

    /// Fixed semantic colors per trigger kind (consistent across the app:
    /// promote = green like the promotions cell, manual = blue, periodic
    /// = gray, signal saves = orange).
    private func badgeColor(_ trigger: String) -> Color {
        switch trigger {
        case "manual": return .blue
        case "periodic": return .secondary
        case "promote": return .green
        case "sigusr2": return .orange
        default: return .purple
        }
    }
}
