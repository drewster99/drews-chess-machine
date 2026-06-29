//
//  HumanPlayPlyReadoutTests.swift
//  DrewsChessMachineTests
//
//  Regression guard for the human-play status-bar "Ply" readout.
//
//  Bug: after a Revert, the status bar's "Ply" dropped below the move
//  list. The move list, board, and live counter are all seeded with the
//  reverted-in prefix, but the post-game readout used
//  `GameStats.totalMoves`, which counts only the plies the most recent
//  `ChessMachine.runGameLoop` played after the revert — excluding the
//  seeded prefix. The readout must instead reflect the displayed game
//  length (the move-list `history` count), in every state.
//

import XCTest
@testable import DrewsChessMachine

final class HumanPlayPlyReadoutTests: XCTestCase {

    /// Reverted game: an 81-ply kept prefix plus 53 plies played after
    /// the revert is 134 plies on screen, but the post-revert
    /// `runGameLoop` only counted 53 (`GameStats.totalMoves`). The
    /// readout must show 134 — the move-list length — not 53.
    ///
    /// Under the old logic (`engineLoopTotalMoves` when present) this
    /// returns 53 and fails; that is the regression this test pins.
    func testReadoutUsesFullHistoryAfterRevert() {
        XCTAssertEqual(
            HumanPlayPlyReadout.displayedPlyCount(
                displayedHistoryPlies: 134,
                engineLoopTotalMoves: 53
            ),
            134
        )
    }

    /// Non-reverted finished game: the displayed length and the engine's
    /// per-loop total agree, so the readout is unchanged.
    func testReadoutMatchesWhenNotReverted() {
        XCTAssertEqual(
            HumanPlayPlyReadout.displayedPlyCount(
                displayedHistoryPlies: 40,
                engineLoopTotalMoves: 40
            ),
            40
        )
    }

    /// Mid-game (no `GameStats` yet): still reflects the displayed
    /// history length.
    func testReadoutInProgressUsesHistory() {
        XCTAssertEqual(
            HumanPlayPlyReadout.displayedPlyCount(
                displayedHistoryPlies: 7,
                engineLoopTotalMoves: nil
            ),
            7
        )
    }
}
