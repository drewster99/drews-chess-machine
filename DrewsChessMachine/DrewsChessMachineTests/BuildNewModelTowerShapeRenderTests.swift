//
//  BuildNewModelTowerShapeRenderTests.swift
//  DrewsChessMachineTests
//
//  The Build New Model screen with a tower shape the user typed but that
//  cannot be built: a negative block count, a count of `Int.max`, or a tower
//  whose training state cannot fit in any Mac's memory. The screen once
//  trapped on all three while drawing (the init-options rows asked for the
//  group's skip projection, which expanded the tower block by block; the
//  "Total blocks" readout summed the counts without overflow checking), and
//  the "Neutral init" button trapped the same way. These tests host the real
//  screen, type the count through the group's draft, and let it redraw.
//

import AppKit
import SwiftUI
import XCTest
@testable import DrewsChessMachine

@MainActor
final class BuildNewModelTowerShapeRenderTests: XCTestCase {

    /// Three single-block groups of distinct widths, so the second and third
    /// groups have skip projections and their init-option rows are drawn.
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

    /// Hosts the screen on a three-group model, sets the middle group's
    /// block count to `count`, lets the screen redraw, and returns the
    /// screen's validation error.
    private func validationErrorAfterSettingMiddleGroupCount(to count: Int) -> String? {
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

        model.blockGroupDrafts[1].group.count = count
        settle(host)
        XCTAssertNil(model.buildRequest, "Build must be disabled for a tower that cannot be built")
        return model.validationError
    }

    func testANegativeBlockCountIsShownAsAnError() throws {
        let error = try XCTUnwrap(validationErrorAfterSettingMiddleGroupCount(to: -1))
        XCTAssertTrue(error.contains("blockGroups[1].count"), error)
    }

    func testAnIntMaxBlockCountIsShownAsAnError() throws {
        let error = try XCTUnwrap(validationErrorAfterSettingMiddleGroupCount(to: Int.max))
        XCTAssertTrue(error.contains("overflow"), error)
    }

    func testATowerTooLargeForThisMacIsShownAsAnError() throws {
        let error = try XCTUnwrap(validationErrorAfterSettingMiddleGroupCount(to: 1 << 44))
        XCTAssertTrue(error.contains("physical memory"), error)
    }

    func testNeutralInitWithANegativeBlockCountKeepsTheError() throws {
        let model = threeGroupModel()
        model.blockGroupDrafts[1].group.count = -1
        model.applyNeutralInit()
        let error = try XCTUnwrap(model.validationError)
        XCTAssertTrue(error.contains("blockGroups[1].count"), error)
    }
}
