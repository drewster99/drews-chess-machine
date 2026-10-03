//
//  BuildNewModelGroupRemovalRenderTests.swift
//  DrewsChessMachineTests
//
//  Removing a block group under the live Build New Model screen. SwiftUI
//  evaluates a removed row's views once more after its draft has left the
//  model, so nothing a row reads while drawing may require its draft to still
//  be in the tower. That once trapped the app (`BuildNewModelModel.position(of:)`
//  from the init-options row) on any group removal. These tests host the real
//  screen in a window, remove a group through the same model call the row's
//  remove button makes, and let the screen redraw.
//

import AppKit
import SwiftUI
import XCTest
@testable import DrewsChessMachine

@MainActor
final class BuildNewModelGroupRemovalRenderTests: XCTestCase {

    /// Three single-block groups of distinct widths, so the width changes
    /// give the second and third groups skip projections (their init-option
    /// rows are on screen) and each group is recognizable in the result.
    private func threeGroupModel() -> BuildNewModelModel {
        var arch = NetworkArchitecture.current
        let base = arch.blockGroups[0]
        arch.blockGroups = [16, 32, 64].map { width in
            var group = base
            group.count = 1
            group.channels = width
            return group
        }
        return BuildNewModelModel(NamedArchitecture(label: "test", architecture: arch))
    }

    /// Lets SwiftUI finish the update a model change scheduled.
    private func settle(_ host: NSView) {
        host.layoutSubtreeIfNeeded()
        RunLoop.main.run(until: Date().addingTimeInterval(0.3))
        host.layoutSubtreeIfNeeded()
    }

    /// Hosts the Build New Model screen on `model` in an on-screen window tall
    /// enough that every group row is realized, removes the group at
    /// `removedPosition`, lets the screen redraw, and returns the widths left.
    private func widthsAfterRemovingGroup(at removedPosition: Int) -> [Int] {
        let model = threeGroupModel()
        let view = BuildNewModelView(model: model, onBuild: { _ in }, onCancel: {})
        let host = NSHostingView(rootView: view.frame(width: 1000, height: 6000))
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1000, height: 6000),
            styleMask: [.titled],
            backing: .buffered,
            defer: false
        )
        window.isReleasedWhenClosed = false
        window.contentView = host
        window.orderFrontRegardless()
        defer {
            window.orderOut(nil)
            window.contentView = nil
        }
        settle(host)

        model.removeGroup(model.blockGroupDrafts[removedPosition])
        settle(host)
        return model.architecture.blockGroups.map(\.channels)
    }

    func testRemovingTheMiddleGroupKeepsTheOtherTwo() {
        XCTAssertEqual(widthsAfterRemovingGroup(at: 1), [16, 64])
    }

    func testRemovingTheLastGroupKeepsTheFirstTwo() {
        XCTAssertEqual(widthsAfterRemovingGroup(at: 2), [16, 32])
    }

    func testRemovingTheFirstGroupKeepsTheLastTwo() {
        XCTAssertEqual(widthsAfterRemovingGroup(at: 0), [32, 64])
    }
}
