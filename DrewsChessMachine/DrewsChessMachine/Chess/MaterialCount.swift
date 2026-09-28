import Foundation

/// Each side's material in standard points (pawn 1, knight 3, bishop 3,
/// rook 5, queen 9; kings count 0), from `PieceType.materialValue`.
struct MaterialCount: Sendable, Equatable {
    let white: Int
    let black: Int

    init(white: Int, black: Int) {
        self.white = white
        self.black = black
    }

    init(_ state: GameState) {
        var white = 0
        var black = 0
        for case let piece? in state.board {
            switch piece.color {
            case .white: white += piece.type.materialValue
            case .black: black += piece.type.materialValue
            }
        }
        self.init(white: white, black: black)
    }

    func points(for color: PieceColor) -> Int {
        color == .white ? white : black
    }

    /// `color`'s points minus the other side's: positive when ahead.
    func lead(for color: PieceColor) -> Int {
        color == .white ? white - black : black - white
    }
}
