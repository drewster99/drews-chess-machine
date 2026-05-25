import AppKit
import SceneKit
import SwiftUI

// MARK: - Surface metric

/// Which per-ply quantity the arena surface plots as its height and
/// color. Each metric is bucketed by absolute ply and stacked across
/// arenas; the user picks one with a segmented control.
enum SurfaceMetric: String, CaseIterable, Identifiable {
    /// Candidate win rate `(W + 0.5·D)/N` — the arena's own scoring
    /// identity, per ply bucket.
    case winRate = "Win rate"
    /// Mean value-head scalar `p_win − p_loss` — what the candidate
    /// network *thought* of its position, per ply bucket.
    case valueHead = "Value head"
    /// Mean candidate material advantage — standard piece values,
    /// candidate minus opponent — per ply bucket.
    case materialAdvantage = "Material advantage"

    var id: String { rawValue }

    /// `[lo, hi]` clamp window for the surface's height and color. The
    /// interesting band is narrow for every metric (a near-parity
    /// candidate sits at the neutral midpoint), so clamping to this
    /// window gives the band visible relief instead of flattening
    /// everything toward the middle.
    var displayRange: (lo: Double, hi: Double) {
        switch self {
        case .winRate:           return (0.40, 0.60)
        case .valueHead:         return (-0.10, 0.10)
        case .materialAdvantage: return (-2.0, 2.0)
        }
    }

    /// The neutral midpoint — rendered gray on the diverging color map,
    /// and used as the fill value for cells past an arena's longest
    /// game (`0.5` is the all-draw win rate; `0.0` is both the even
    /// value scalar and the even material advantage).
    var neutralValue: Double {
        switch self {
        case .winRate:                       return 0.5
        case .valueHead, .materialAdvantage: return 0.0
        }
    }
}

// MARK: - Data aggregation

/// The per-ply chart, stacked across every prior arena into a 3D
/// surface: each arena contributes one row of samples and the rows are
/// laid out along a Z axis in chronological order, so the user sees the
/// candidate's per-ply profile *evolve* over the training session at a
/// glance.
///
/// Two complications drive the re-binning logic below:
///
///   1. Bucket-width drift. Some persisted arena summaries used 5-ply
///      buckets while current arenas use 20-ply buckets
///      (`ArenaSummaryAggregator.plyBucketWidth`). A surface
///      needs every row on a common X grid, so each source bucket is
///      mapped to a common 20-ply column by `lowerInclusive / 20`.
///      Integer division collapses four consecutive legacy 5-ply
///      buckets (0-4, 5-9, 10-14, 15-19) onto common column 0, and a
///      native 20-ply bucket (lower 0) onto the same column — both
///      widths land correctly with no width-specific branching.
///
///   2. Ragged game lengths. A short-game arena has no buckets past,
///      say, ply 60 while a long-game arena reaches ply 200. The grid
///      must stay rectangular for the mesh, so absent (arena, column)
///      cells are filled with the metric's neutral value, which reads
///      as neutral gray on the diverging color map.
struct ArenaSurfaceGrid {
    /// Which metric `values` carries — drives the height/color mapping.
    let metric: SurfaceMetric
    /// `values[arenaRow][plyColumn]`, fully rectangular: `rowCount`
    /// rows by `columnCount` columns. Row 0 is the oldest arena.
    let values: [[Double]]
    /// Number of arena rows (Z axis). Equals `values.count`.
    let rowCount: Int
    /// Number of common 20-ply columns (X axis).
    let columnCount: Int

    /// Common ply-grid column width, in plies. All source buckets are
    /// re-binned onto multiples of this regardless of their original
    /// width.
    static let commonColumnWidth = 20

    /// Ply midpoint represented by column `c`, used for axis captions.
    static func plyMidpoint(forColumn c: Int) -> Int {
        commonColumnWidth * c + commonColumnWidth / 2
    }

    /// Build the rectangular grid for `metric` from a chronological
    /// arena history. Only records that carry a non-empty `valueByPly`
    /// breakdown become rows; records without an `extendedSummary` (or
    /// with an empty one) are skipped entirely so the Z axis stays
    /// contiguous rather than leaving gaps for un-summarized arenas.
    static func build(history: [TournamentRecord], metric: SurfaceMetric) -> ArenaSurfaceGrid {
        // Re-bin each qualifying arena onto the common 20-ply grid.
        // Win rate sums raw W/D/L (summing the counts, not averaging
        // pre-computed scores, keeps the merged win rate exact). Value
        // head and material advantage sum `mean · count` so the
        // re-binned value is the count-weighted mean over the merged
        // sample population — averaging the pre-computed per-bucket
        // means directly would be wrong whenever the merged buckets
        // differ in sample count.
        struct ColumnTally {
            var wins = 0
            var draws = 0
            var losses = 0
            var valueWeightedSum = 0.0
            var materialWeightedSum = 0.0
        }

        var perArenaColumns: [[Int: ColumnTally]] = []
        var maxCommonColumn = -1

        for record in history {
            guard let summary = record.extendedSummary,
                  !summary.valueByPly.isEmpty else {
                continue
            }
            var columns: [Int: ColumnTally] = [:]
            for bucket in summary.valueByPly {
                // Integer division maps both 5-ply and 20-ply source
                // buckets onto the same common column index.
                let commonColumn = bucket.lowerInclusive / commonColumnWidth
                var tally = columns[commonColumn] ?? ColumnTally()
                tally.wins += bucket.wins
                tally.draws += bucket.draws
                tally.losses += bucket.losses
                tally.valueWeightedSum += Double(bucket.mean) * Double(bucket.count)
                tally.materialWeightedSum +=
                    Double(bucket.meanMaterialAdvantage ?? 0) * Double(bucket.count)
                columns[commonColumn] = tally
                maxCommonColumn = max(maxCommonColumn, commonColumn)
            }
            perArenaColumns.append(columns)
        }

        // No qualifying arenas — an empty grid; the host view renders a
        // placeholder when the row/column counts are too small.
        guard maxCommonColumn >= 0, !perArenaColumns.isEmpty else {
            return ArenaSurfaceGrid(metric: metric, values: [], rowCount: 0, columnCount: 0)
        }

        let columnCount = maxCommonColumn + 1
        let rowCount = perArenaColumns.count
        let baseline = metric.neutralValue

        // Materialize the rectangular grid. Cells with no source bucket
        // (or a zero-sample tally) fall back to the neutral baseline.
        var values: [[Double]] = []
        values.reserveCapacity(rowCount)
        for columns in perArenaColumns {
            var row: [Double] = []
            row.reserveCapacity(columnCount)
            for c in 0..<columnCount {
                guard let tally = columns[c] else {
                    row.append(baseline)
                    continue
                }
                let n = tally.wins + tally.draws + tally.losses
                guard n > 0 else {
                    row.append(baseline)
                    continue
                }
                switch metric {
                case .winRate:
                    row.append((Double(tally.wins) + 0.5 * Double(tally.draws)) / Double(n))
                case .valueHead:
                    row.append(tally.valueWeightedSum / Double(n))
                case .materialAdvantage:
                    row.append(tally.materialWeightedSum / Double(n))
                }
            }
            values.append(row)
        }

        return ArenaSurfaceGrid(
            metric: metric,
            values: values,
            rowCount: rowCount,
            columnCount: columnCount
        )
    }
}

// MARK: - SceneKit surface

/// `NSViewRepresentable` wrapping an `SCNView` that renders the surface
/// grid as a height-mapped, per-vertex-colored triangle mesh.
///
/// The mesh uses `lightingModel = .constant`: the surface carries no
/// normals and needs none — each vertex's metric value maps directly to
/// both its height and its color, and constant shading paints that
/// vertex color through unmodified by any light. This keeps the
/// geometry construction to two buffers (positions, colors) plus an
/// index buffer with no normal computation.
struct SceneKitSurfaceView: NSViewRepresentable {
    let grid: ArenaSurfaceGrid

    func makeNSView(context: Context) -> SCNView {
        let scnView = SCNView()
        Self.configure(scnView)
        installScene(into: scnView)
        context.coordinator.lastGrid = grid
        return scnView
    }

    /// One-time view configuration. `backgroundColor` is assigned the
    /// explicitly-typed `NSColor.clear` rather than a bare `.clear`:
    /// `SCNView.backgroundColor` imports without a concrete element
    /// type, so a contextually-inferred `.clear` would push the
    /// type-checker into an expensive overload search.
    private static func configure(_ scnView: SCNView) {
        scnView.allowsCameraControl = true
        scnView.autoenablesDefaultLighting = true
        // Clear background so the sheet's material shows through the
        // empty space around the surface.
        scnView.backgroundColor = NSColor.clear
    }

    func updateNSView(_ scnView: SCNView, context: Context) {
        // Rebuilding the whole scene is cheap relative to the rarity of
        // a grid change (a new arena completes, or the user toggles the
        // metric) and avoids having to diff vertex buffers.
        guard !gridsEqual(context.coordinator.lastGrid, grid) else { return }
        installScene(into: scnView)
        context.coordinator.lastGrid = grid
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    /// Holds the last grid rendered so `updateNSView` can skip a
    /// redundant rebuild when SwiftUI re-invokes it with no real
    /// change.
    final class Coordinator {
        var lastGrid: ArenaSurfaceGrid?
    }

    // MARK: Scene construction

    /// Construct an `SCNVector3` from `Double` components. Spelled out
    /// rather than relying on literal type inference at each call site:
    /// `SCNVector3`'s macOS initializer takes `CGFloat`, and inline
    /// numeric literals forced the Swift type-checker into an
    /// expensive overload-resolution pass. Funneling every
    /// construction through this typed helper keeps those expressions
    /// trivial.
    private static func vector3(_ x: Double, _ y: Double, _ z: Double) -> SCNVector3 {
        SCNVector3(CGFloat(x), CGFloat(y), CGFloat(z))
    }

    /// Center of the unit-ish surface, used as the camera's look-at
    /// target. The surface spans x,z ∈ [0, 2]; the y component is a
    /// mid-height so the camera frames the mesh body rather than its
    /// floor.
    private static let surfaceCenter = vector3(1.0, 0.3, 1.0)

    /// Build a fresh scene from the current grid and install it into
    /// the view, also pointing the view's camera at the new 3/4-view
    /// camera node. Both `makeNSView` and `updateNSView` route through
    /// here so the `pointOfView` is always wired to the live scene.
    ///
    /// The scene assembly is deliberately fanned out across the small
    /// `make*` helpers below — keeping each helper down to one or two
    /// SceneKit types each keeps every body trivial for the type-checker.
    private func installScene(into scnView: SCNView) {
        let scene = makeBaseScene()
        addSurfaceNodeIfPossible(to: scene)
        addAxisLabels(to: scene)
        let targetNode = addLookAtTarget(to: scene)
        let cameraNode = addCamera(to: scene, aimedAt: targetNode)
        scnView.scene = scene
        scnView.pointOfView = cameraNode
    }

    /// An empty scene — the assembly's starting point.
    private func makeBaseScene() -> SCNScene {
        SCNScene()
    }

    /// Add text labels oriented along each axis: "Ply" and "Arena" laid
    /// flat on the XZ ground plane so each genuinely runs *along* its
    /// own axis, plus the metric name upright above the Y (height) axis.
    /// They are ordinary scene nodes — no billboard constraint — so they
    /// rotate and translate with the surface as the user orbits the
    /// camera, reading as part of the 3D scene. Skipped for a degenerate
    /// grid (the host shows a placeholder instead).
    ///
    /// `SCNText` is built upright in its local XY plane with glyphs
    /// running along local +X. To lay a label flat on the floor it is
    /// pitched −90° about local X (tipping the upright text down, face
    /// up). The Ply axis runs along world +X, so that single pitch is
    /// all Ply needs. The Arena axis runs along world Z, so the flat
    /// Arena label is additionally nested in a holder yawed +90° about
    /// Y — composing two single-axis rotations rather than one
    /// multi-axis `eulerAngles`, whose component order is easy to get
    /// wrong. After the holder yaw the glyphs run along world −Z, so the
    /// Arena string leads with a "←" — which then points along +Z
    /// (oldest→newest) — and its letters stay upright to the default
    /// 3/4 camera.
    private func addAxisLabels(to scene: SCNScene) {
        guard grid.rowCount >= 2, grid.columnCount >= 2 else { return }

        // Ply: flat on the floor, glyphs running along world +X.
        let plyLabel = makeAxisLabel("Ply →")
        plyLabel.eulerAngles = Self.vector3(-.pi / 2, 0.0, 0.0)
        plyLabel.position = Self.vector3(2.25, 0.0, 1.0)
        scene.rootNode.addChildNode(plyLabel)

        // Arena: flat on the floor, glyphs running along world Z. The
        // label is pitched flat, then a holder yaws it onto the Z axis.
        let arenaLabel = makeAxisLabel("← Arena")
        arenaLabel.eulerAngles = Self.vector3(-.pi / 2, 0.0, 0.0)
        let arenaHolder = SCNNode()
        arenaHolder.addChildNode(arenaLabel)
        arenaHolder.eulerAngles = Self.vector3(0.0, .pi / 2, 0.0)
        arenaHolder.position = Self.vector3(1.0, 0.0, 2.25)
        scene.rootNode.addChildNode(arenaHolder)

        // Metric name: upright caption above the Y (height) axis.
        let metricLabel = makeAxisLabel(grid.metric.rawValue)
        metricLabel.position = Self.vector3(-0.3, 1.2, -0.3)
        scene.rootNode.addChildNode(metricLabel)
    }

    /// One axis label: flat `SCNText`, scaled from font points down
    /// into the 2-unit scene and pivoted on its bounding-box center so
    /// the node's position is the label's center. No billboard
    /// constraint — the label is a fixed part of the scene and orbits
    /// with the surface. Sizing/placement here is tuned by eye and may
    /// want nudging.
    private func makeAxisLabel(_ text: String) -> SCNNode {
        let scnText = SCNText(string: text, extrusionDepth: 0.0)
        scnText.font = NSFont.systemFont(ofSize: 12)
        scnText.flatness = 0.1
        let material = SCNMaterial()
        material.lightingModel = .constant
        material.diffuse.contents = NSColor.secondaryLabelColor
        material.isDoubleSided = true
        scnText.materials = [material]

        let node = SCNNode(geometry: scnText)
        node.scale = Self.vector3(0.015, 0.015, 0.015)
        // SCNText's origin is its bounding-box corner; re-pivot to the
        // center so `position` places the label's middle on the axis.
        let box = scnText.boundingBox
        node.pivot = SCNMatrix4MakeTranslation(
            (box.min.x + box.max.x) / 2,
            (box.min.y + box.max.y) / 2,
            (box.min.z + box.max.z) / 2
        )
        return node
    }

    /// Build the surface mesh from the current grid and attach it. A
    /// degenerate grid yields no node; the host view gates on
    /// row/column counts before ever showing this representable, so a
    /// missing surface here is belt-and-suspenders.
    private func addSurfaceNodeIfPossible(to scene: SCNScene) {
        guard grid.rowCount >= 2, grid.columnCount >= 2,
              let geometry = makeSurfaceGeometry(grid: grid) else {
            return
        }
        let surfaceNode = SCNNode(geometry: geometry)
        scene.rootNode.addChildNode(surfaceNode)
    }

    /// Add (and return) the empty node the camera's look-at constraint
    /// aims at. `SCNLookAtConstraint` targets a node, not a raw point,
    /// so the surface-center coordinate must live in the scene graph.
    private func addLookAtTarget(to scene: SCNScene) -> SCNNode {
        let targetNode = SCNNode()
        targetNode.position = Self.surfaceCenter
        scene.rootNode.addChildNode(targetNode)
        return targetNode
    }

    /// Add (and return) the 3/4-view camera node, constrained to keep
    /// the surface center framed as the user orbits.
    private func addCamera(to scene: SCNScene, aimedAt targetNode: SCNNode) -> SCNNode {
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        // 3/4 view: above and diagonally off one corner of the surface.
        cameraNode.position = Self.vector3(3.5, 3.0, 3.5)
        let lookAt = SCNLookAtConstraint(target: targetNode)
        cameraNode.constraints = [lookAt]
        scene.rootNode.addChildNode(cameraNode)
        return cameraNode
    }

    /// Build the height-mapped triangle mesh from the grid.
    ///
    /// Vertex layout is row-major: vertex index for (r, c) is
    /// `r * columnCount + c`. Each interior cell emits two triangles
    /// covering its four corner vertices. Winding is irrelevant because
    /// the material is double-sided, so the two triangles are emitted
    /// in a fixed order without back-face concern.
    private func makeSurfaceGeometry(grid: ArenaSurfaceGrid) -> SCNGeometry? {
        let rows = grid.rowCount
        let cols = grid.columnCount
        guard rows >= 2, cols >= 2 else { return nil }

        var positions: [SCNVector3] = []
        positions.reserveCapacity(rows * cols)
        var colors: [SCNVector4] = []
        colors.reserveCapacity(rows * cols)

        // Denominators clamped to >= 1 so a single-row / single-column
        // grid can't divide by zero (the host gates this off anyway).
        let colSpan = Float(max(cols - 1, 1))
        let rowSpan = Float(max(rows - 1, 1))

        for r in 0..<rows {
            let row = grid.values[r]
            for c in 0..<cols {
                let value = row[c]
                let x: Float = Float(c) / colSpan * 2.0
                let z: Float = Float(r) / rowSpan * 2.0
                let y: Float = Float(heightNorm(value: value, metric: grid.metric)) * 1.0
                positions.append(Self.vector3(Double(x), Double(y), Double(z)))

                colors.append(divergingColor(value: value, metric: grid.metric))
            }
        }

        // Two triangles per interior cell.
        var indices: [Int32] = []
        indices.reserveCapacity((rows - 1) * (cols - 1) * 6)
        for r in 0..<(rows - 1) {
            for c in 0..<(cols - 1) {
                let topLeft = Int32(r * cols + c)
                let topRight = Int32(r * cols + (c + 1))
                let bottomLeft = Int32((r + 1) * cols + c)
                let bottomRight = Int32((r + 1) * cols + (c + 1))
                // Triangle 1: TL, TR, BL. Triangle 2: TR, BR, BL.
                indices.append(contentsOf: [topLeft, topRight, bottomLeft])
                indices.append(contentsOf: [topRight, bottomRight, bottomLeft])
            }
        }

        let positionSource = SCNGeometrySource(vertices: positions)
        let colorGeometrySource = colorSource(from: colors)
        let element = SCNGeometryElement(indices: indices, primitiveType: .triangles)

        let geometry = SCNGeometry(
            sources: [positionSource, colorGeometrySource],
            elements: [element]
        )

        let material = SCNMaterial()
        // Constant lighting paints the per-vertex color directly — no
        // normals, no light interaction needed for a data surface.
        material.lightingModel = .constant
        // Winding is unmanaged above; double-siding makes both faces
        // visible from any camera angle the user orbits to.
        material.isDoubleSided = true
        geometry.materials = [material]

        return geometry
    }

    /// Build a per-vertex color `SCNGeometrySource`. There is no
    /// vertices-style convenience initializer for color, so the RGBA
    /// floats are packed into `Data` and described explicitly.
    private func colorSource(from colors: [SCNVector4]) -> SCNGeometrySource {
        var floats: [Float] = []
        floats.reserveCapacity(colors.count * 4)
        for color in colors {
            floats.append(Float(color.x))
            floats.append(Float(color.y))
            floats.append(Float(color.z))
            floats.append(Float(color.w))
        }
        let data = floats.withUnsafeBufferPointer { Data(buffer: $0) }
        let componentStride = MemoryLayout<Float>.size
        let vectorStride = componentStride * 4
        return SCNGeometrySource(
            data: data,
            semantic: .color,
            vectorCount: colors.count,
            usesFloatComponents: true,
            componentsPerVector: 4,
            bytesPerComponent: componentStride,
            dataOffset: 0,
            dataStride: vectorStride
        )
    }

    // MARK: Value → geometry mappings

    /// Map a metric value to a normalized height in [0, 1] by clamping
    /// to the metric's `displayRange`. The interesting band is narrow
    /// for both metrics, so the clamp gives the surface visible relief
    /// where promotion decisions actually live.
    private func heightNorm(value: Double, metric: SurfaceMetric) -> Double {
        let (lo, hi) = metric.displayRange
        let clamped = min(max(value, lo), hi)
        return (clamped - lo) / (hi - lo)
    }

    /// Diverging red→gray→green color for a metric value. Red marks the
    /// candidate below the metric's neutral midpoint, gray marks
    /// parity, green marks above; the color saturates a quarter of the
    /// display window out from neutral and interpolates within that
    /// quarter. Returned as an RGBA vector for direct packing into the
    /// color geometry source.
    private func divergingColor(value: Double, metric: SurfaceMetric) -> SCNVector4 {
        let red = NSColor(calibratedRed: 0.85, green: 0.20, blue: 0.20, alpha: 1.0)
        let gray = NSColor(calibratedRed: 0.80, green: 0.80, blue: 0.80, alpha: 1.0)
        let green = NSColor(calibratedRed: 0.20, green: 0.75, blue: 0.30, alpha: 1.0)

        let (lo, hi) = metric.displayRange
        // Full red/green a quarter of the display window out from the
        // neutral midpoint; interpolate across that inner quarter.
        let saturation = (hi - lo) / 4.0
        let delta = value - metric.neutralValue

        let color: NSColor
        if delta <= -saturation {
            color = red
        } else if delta < 0 {
            let t = CGFloat((delta + saturation) / saturation)
            color = interpolate(from: red, to: gray, t: t)
        } else if delta >= saturation {
            color = green
        } else {
            let t = CGFloat(delta / saturation)
            color = interpolate(from: gray, to: green, t: t)
        }
        return SCNVector4(
            CGFloat(color.redComponent),
            CGFloat(color.greenComponent),
            CGFloat(color.blueComponent),
            CGFloat(color.alphaComponent)
        )
    }

    /// Linear RGBA interpolation between two calibrated-RGB colors.
    private func interpolate(from: NSColor, to: NSColor, t: CGFloat) -> NSColor {
        let clampedT = min(max(t, 0), 1)
        return NSColor(
            calibratedRed: from.redComponent + (to.redComponent - from.redComponent) * clampedT,
            green: from.greenComponent + (to.greenComponent - from.greenComponent) * clampedT,
            blue: from.blueComponent + (to.blueComponent - from.blueComponent) * clampedT,
            alpha: from.alphaComponent + (to.alphaComponent - from.alphaComponent) * clampedT
        )
    }

    /// Two surface grids are equal for redraw purposes iff they carry
    /// the same metric and every cell matches. Metric and dimensions
    /// are compared first as a fast reject.
    private func gridsEqual(_ lhs: ArenaSurfaceGrid?, _ rhs: ArenaSurfaceGrid) -> Bool {
        guard let lhs else { return false }
        guard lhs.metric == rhs.metric,
              lhs.rowCount == rhs.rowCount,
              lhs.columnCount == rhs.columnCount else {
            return false
        }
        return lhs.values == rhs.values
    }
}

// MARK: - Host view

/// Sheet content hosting the 3D arena surface. Builds the re-binned
/// grid from the session's arena history for the selected metric and
/// either renders the SceneKit surface or, when there isn't enough
/// data for a meaningful mesh, a centered placeholder.
struct ArenaSurfaceView: View {
    let history: [TournamentRecord]
    let onClose: () -> Void

    /// Metric the surface currently plots; toggled by the header's
    /// segmented control.
    @State private var metric: SurfaceMetric = .winRate

    var body: some View {
        // Re-bin the selected metric once per render. Cheap (a handful
        // of integer-bucket passes), but both the renderability gate
        // and the SceneKit view need it, so build it here rather than
        // in a computed property that would run twice.
        let grid = ArenaSurfaceGrid.build(history: history, metric: metric)
        // A surface needs at least a 2×2 grid — two arenas to span the
        // Z axis and two ply columns to span the X axis — before there
        // are any triangles to draw.
        let hasRenderableSurface = grid.rowCount >= 2 && grid.columnCount >= 2
        return VStack(spacing: 0) {
            header
            Divider()
            if hasRenderableSurface {
                SceneKitSurfaceView(grid: grid)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                caption
            } else {
                placeholder
            }
        }
        .frame(
            minWidth: 960, idealWidth: 1230,
            minHeight: 720, idealHeight: 930
        )
    }

    @ViewBuilder
    private var header: some View {
        HStack {
            Text("Arena surface")
                .font(.title2.weight(.semibold))
            Spacer()
            Picker("Metric", selection: $metric) {
                ForEach(SurfaceMetric.allCases) { option in
                    Text(option.rawValue).tag(option)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()
            Button("Close", action: onClose)
                .keyboardShortcut(.cancelAction)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    @ViewBuilder
    private var caption: some View {
        Text(captionText)
            .font(.caption)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
    }

    /// Axis legend, worded for the selected metric.
    private var captionText: String {
        switch metric {
        case .winRate:
            return "X → ply (early→late) · Z → arena (oldest→newest) · height & color → candidate win rate (red <0.5, green >0.5). Cells past an arena's longest game are filled at 0.50."
        case .valueHead:
            return "X → ply (early→late) · Z → arena (oldest→newest) · height & color → mean value-head scalar p_win − p_loss (red <0, green >0). Cells past an arena's longest game are filled at 0.00."
        case .materialAdvantage:
            return "X → ply (early→late) · Z → arena (oldest→newest) · height & color → mean candidate material advantage in piece-value points (red <0, green >0). Cells past an arena's longest game are filled at 0.00."
        }
    }

    @ViewBuilder
    private var placeholder: some View {
        VStack {
            Spacer()
            Text("Need at least two arenas with breakdown data to render a surface.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
