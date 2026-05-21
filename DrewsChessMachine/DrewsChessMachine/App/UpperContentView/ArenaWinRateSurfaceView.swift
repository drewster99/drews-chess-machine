import AppKit
import SceneKit
import SwiftUI

// MARK: - Data aggregation

/// The win-rate-by-ply chart, stacked across every prior arena into a
/// 3D surface: each arena contributes one row of win-rate samples and
/// the rows are laid out along a Z axis in chronological order, so the
/// user sees the candidate's per-ply scoring profile *evolve* over the
/// training session at a glance.
///
/// Two complications drive the re-binning logic below:
///
///   1. Bucket-width drift. Arenas run before a recent change emitted
///      `valueByPly` in 5-ply buckets; current arenas use 20-ply
///      buckets (`ArenaSummaryAggregator.plyBucketWidth`). A surface
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
///      cells are filled with 0.5 — the draw / even-match baseline,
///      which reads as neutral gray on the diverging color map rather
///      than as a spurious win or loss.
struct ArenaWinRateSurfaceGrid {
    /// `winRate[arenaRow][plyColumn]`, fully rectangular: `rowCount`
    /// rows by `columnCount` columns. Row 0 is the oldest arena.
    let winRate: [[Double]]
    /// Number of arena rows (Z axis). Equals `winRate.count`.
    let rowCount: Int
    /// Number of common 20-ply columns (X axis).
    let columnCount: Int

    /// Common ply-grid column width, in plies. All source buckets are
    /// re-binned onto multiples of this regardless of their original
    /// width.
    static let commonColumnWidth = 20

    /// Win-rate value used for (arena, column) cells past an arena's
    /// longest game. 0.5 is the even-match / all-draw score, which the
    /// diverging color map renders as neutral gray.
    static let baselineWinRate = 0.5

    /// Ply midpoint represented by column `c`, used for axis captions.
    static func plyMidpoint(forColumn c: Int) -> Int {
        commonColumnWidth * c + commonColumnWidth / 2
    }

    /// Build the rectangular win-rate grid from a chronological arena
    /// history. Only records that carry a non-empty `valueByPly`
    /// breakdown become rows; records without an `extendedSummary`
    /// (or with an empty one) are skipped entirely so the Z axis stays
    /// contiguous rather than leaving gaps for un-summarized arenas.
    static func build(history: [TournamentRecord]) -> ArenaWinRateSurfaceGrid {
        // (1) Re-bin each qualifying arena onto the common 20-ply grid,
        // summing raw W/D/L across every source bucket that maps to the
        // same common column. Summing the counts (rather than averaging
        // pre-computed scores) keeps the re-binned win rate exact: it's
        // still (W + 0.5·D)/N over the merged sample population.
        struct ColumnTally {
            var wins = 0
            var draws = 0
            var losses = 0
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
                columns[commonColumn] = tally
                maxCommonColumn = max(maxCommonColumn, commonColumn)
            }
            perArenaColumns.append(columns)
        }

        // No qualifying arenas — an empty grid; the host view renders a
        // placeholder when the row/column counts are too small.
        guard maxCommonColumn >= 0, !perArenaColumns.isEmpty else {
            return ArenaWinRateSurfaceGrid(winRate: [], rowCount: 0, columnCount: 0)
        }

        let columnCount = maxCommonColumn + 1
        let rowCount = perArenaColumns.count

        // (2) Materialize the rectangular grid. Cells with no source
        // bucket for that arena fall back to the neutral baseline.
        var winRate: [[Double]] = []
        winRate.reserveCapacity(rowCount)
        for columns in perArenaColumns {
            var row: [Double] = []
            row.reserveCapacity(columnCount)
            for c in 0..<columnCount {
                if let tally = columns[c] {
                    let n = tally.wins + tally.draws + tally.losses
                    if n > 0 {
                        let score = (Double(tally.wins) + 0.5 * Double(tally.draws)) / Double(n)
                        row.append(score)
                    } else {
                        row.append(baselineWinRate)
                    }
                } else {
                    row.append(baselineWinRate)
                }
            }
            winRate.append(row)
        }

        return ArenaWinRateSurfaceGrid(
            winRate: winRate,
            rowCount: rowCount,
            columnCount: columnCount
        )
    }
}

// MARK: - SceneKit surface

/// `NSViewRepresentable` wrapping an `SCNView` that renders the
/// win-rate grid as a height-mapped, per-vertex-colored triangle mesh.
///
/// The mesh uses `lightingModel = .constant`: the surface carries no
/// normals and needs none — each vertex's win rate maps directly to
/// both its height and its color, and constant shading paints that
/// vertex color through unmodified by any light. This keeps the
/// geometry construction to two buffers (positions, colors) plus an
/// index buffer with no normal computation.
struct SceneKitSurfaceView: NSViewRepresentable {
    let grid: ArenaWinRateSurfaceGrid

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
        // a grid change (only when a new arena completes while the
        // sheet is open) and avoids having to diff vertex buffers.
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
        var lastGrid: ArenaWinRateSurfaceGrid?
    }

    // MARK: Scene construction

    /// Construct an `SCNVector3` from `Double` components. Spelled out
    /// rather than relying on literal type inference at each call site:
    /// `SCNVector3`'s macOS initializer takes `CGFloat`, and inline
    /// numeric literals forced the Swift type-checker into an
    /// expensive overload-resolution pass (a "took Nms to type-check"
    /// warning). Funneling every construction through this typed
    /// helper keeps those expressions trivial.
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
    /// `make*` helpers below. SceneKit's imported API surface is large
    /// enough that putting the whole assembly in one method body pushes
    /// the Swift type-checker past its long-body warning threshold;
    /// keeping each helper down to one or two SceneKit types each
    /// keeps every body trivial to check.
    private func installScene(into scnView: SCNView) {
        let scene = makeBaseScene()
        addSurfaceNodeIfPossible(to: scene)
        let targetNode = addLookAtTarget(to: scene)
        let cameraNode = addCamera(to: scene, aimedAt: targetNode)
        scnView.scene = scene
        scnView.pointOfView = cameraNode
    }

    /// An empty scene — the assembly's starting point.
    private func makeBaseScene() -> SCNScene {
        SCNScene()
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
    private func makeSurfaceGeometry(grid: ArenaWinRateSurfaceGrid) -> SCNGeometry? {
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
            let row = grid.winRate[r]
            for c in 0..<cols {
                let winRate = row[c]
                let x: Float = Float(c) / colSpan * 2.0
                let z: Float = Float(r) / rowSpan * 2.0
                let y: Float = Float(heightNorm(winRate: winRate)) * 1.0
                positions.append(Self.vector3(Double(x), Double(y), Double(z)))

                let color = divergingColor(winRate: winRate)
                colors.append(color)
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
        let colorSource = colorSource(from: colors)
        let element = SCNGeometryElement(indices: indices, primitiveType: .triangles)

        let geometry = SCNGeometry(
            sources: [positionSource, colorSource],
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

    /// Map a win rate to a normalized height in [0, 1]. The interesting
    /// range of arena scores is narrow — a candidate near parity sits
    /// around 0.5 — so the height window is clamped to [0.40, 0.60].
    /// That gives the surface visible relief for the band where
    /// promotion decisions actually live instead of flattening
    /// everything toward a mid value.
    private func heightNorm(winRate: Double) -> Double {
        let clamped = min(max(winRate, 0.40), 0.60)
        return (clamped - 0.40) / 0.20
    }

    /// Diverging red→gray→green color for a win rate. Red marks the
    /// candidate losing the bucket (≤ 0.45), light gray marks parity
    /// (0.50), green marks winning (≥ 0.55); values between are
    /// linearly interpolated. Returned as an RGBA vector for direct
    /// packing into the color geometry source.
    private func divergingColor(winRate: Double) -> SCNVector4 {
        let red = NSColor(calibratedRed: 0.85, green: 0.20, blue: 0.20, alpha: 1.0)
        let gray = NSColor(calibratedRed: 0.80, green: 0.80, blue: 0.80, alpha: 1.0)
        let green = NSColor(calibratedRed: 0.20, green: 0.75, blue: 0.30, alpha: 1.0)

        let color: NSColor
        if winRate <= 0.45 {
            color = red
        } else if winRate < 0.50 {
            // Interpolate red → gray across (0.45, 0.50).
            let t = CGFloat((winRate - 0.45) / 0.05)
            color = interpolate(from: red, to: gray, t: t)
        } else if winRate >= 0.55 {
            color = green
        } else {
            // Interpolate gray → green across [0.50, 0.55).
            let t = CGFloat((winRate - 0.50) / 0.05)
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

    /// Two surface grids are equal for redraw purposes iff every
    /// win-rate cell matches. Dimensions are compared first as a fast
    /// reject.
    private func gridsEqual(_ lhs: ArenaWinRateSurfaceGrid?, _ rhs: ArenaWinRateSurfaceGrid) -> Bool {
        guard let lhs else { return false }
        guard lhs.rowCount == rhs.rowCount, lhs.columnCount == rhs.columnCount else {
            return false
        }
        return lhs.winRate == rhs.winRate
    }
}

// MARK: - Host view

/// Sheet content hosting the 3D arena win-rate surface. Builds the
/// re-binned grid from the session's arena history and either renders
/// the SceneKit surface or, when there isn't enough data for a
/// meaningful mesh, a centered placeholder.
struct ArenaWinRateSurfaceView: View {
    let history: [TournamentRecord]
    let onClose: () -> Void

    /// Re-binned win-rate grid. Recomputed when `history` changes;
    /// cheap enough (a handful of integer-bucket passes) to do in a
    /// computed property.
    private var grid: ArenaWinRateSurfaceGrid {
        ArenaWinRateSurfaceGrid.build(history: history)
    }

    /// A surface needs at least a 2×2 grid — two arenas to span the Z
    /// axis and two ply columns to span the X axis — before there are
    /// any triangles to draw.
    private var hasRenderableSurface: Bool {
        grid.rowCount >= 2 && grid.columnCount >= 2
    }

    var body: some View {
        VStack(spacing: 0) {
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
            minWidth: 640, idealWidth: 820,
            minHeight: 480, idealHeight: 620
        )
    }

    @ViewBuilder
    private var header: some View {
        HStack {
            Text("Win-rate surface")
                .font(.title2.weight(.semibold))
            Spacer()
            Button("Close", action: onClose)
                .keyboardShortcut(.cancelAction)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    @ViewBuilder
    private var caption: some View {
        Text("X → ply (early→late) · Z → arena (oldest→newest) · height & color → candidate win rate (red <0.5, green >0.5). Cells past an arena's longest game are filled at 0.50.")
            .font(.caption)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
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
