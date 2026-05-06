import SwiftUI

/// Right-hand column of the main panel: Game-run / Candidate-test
/// picker (only when `realTraining` AND `selfPlayWorkers <= 1`),
/// the inline overlay-label / side-to-move / probe-target row, and
/// the live board itself. Each element of the inline row is
/// independently conditional, so the row collapses to just what is
/// currently relevant (e.g. Game-run mode shows nothing; pure
/// forward-pass shows label + side; candidate test in
/// Play-and-Train shows label + Probe).
struct BoardSideView: View {
    @Binding var playAndTrainBoardMode: PlayAndTrainBoardMode
    var sideToMoveBinding: Binding<PieceColor>
    @Binding var probeNetworkTarget: ProbeNetworkTarget
    let realTraining: Bool
    let workerCount: Int
    let inferenceResultPresent: Bool
    let showForwardPassUI: Bool
    let forwardPassEditable: Bool
    let isCandidateTestActive: Bool
    let overlayLabel: String
    /// The board-with-chevrons content. Pre-built by the parent so
    /// this view stays decoupled from the board's own state-heavy
    /// init signature.
    let board: LiveBoardWithNavigationView

    var body: some View {
        // Game-run mode is only meaningful with a single self-play
        // worker (multi-worker sessions can't show one canonical "live
        // game" — Game-run is a placeholder card in that mode). When
        // workers > 1 we hide both the Board picker (Game run vs
        // Candidate test) and the inline side-to-move / Probe row
        // entirely, since Candidate test becomes the only meaningful
        // mode and there's no need to switch into it. This reclaims
        // two rows of vertical space for the actual board.
        let showBoardPicker = realTraining && workerCount <= 1
        VStack(spacing: 4) {
            if showBoardPicker {
                Picker("Board", selection: $playAndTrainBoardMode) {
                    Text("Game run").tag(PlayAndTrainBoardMode.gameRun)
                    Text("Candidate test").tag(PlayAndTrainBoardMode.candidateTest)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(maxWidth: 240)
            }

            HStack(spacing: 12) {
                if inferenceResultPresent, showForwardPassUI {
                    Text(overlayLabel)
                        .font(.system(.caption, design: .monospaced))
                }
                if forwardPassEditable {
                    Picker("To move", selection: sideToMoveBinding) {
                        Text("White").tag(PieceColor.white)
                        Text("Black").tag(PieceColor.black)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .controlSize(.small)
                    .frame(maxWidth: 110)
                }
                if isCandidateTestActive {
                    Picker("Probe", selection: $probeNetworkTarget) {
                        Text("Candidate").tag(ProbeNetworkTarget.candidate)
                        Text("Champion").tag(ProbeNetworkTarget.champion)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .controlSize(.small)
                    .frame(maxWidth: 150)
                }
            }

            board
        }
        // No maxWidth cap — the inner board has aspectRatio(1, .fit)
        // so it'll stay square at min(width, height). When the upper
        // area grows tall, the board grows along with it; the HStack
        // sibling (MainTextPanel / HoverPolicyOverlay) naturally
        // gets the rest of the horizontal space.
        .frame(minWidth: 320)
    }
}
