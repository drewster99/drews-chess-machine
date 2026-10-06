//
//  BuildNewModelSiteActivationRenderTests.swift
//  DrewsChessMachineTests
//
//  The Build New Model screen's six architecture-level activation pickers,
//  hosted for real: across pre/post first and last groups, simple_conv and
//  the compress fusion on and off, the screen draws without a trap, and each
//  site's picker presentation (`ArchitectureSiteActivationPicker.presentation`,
//  the value the picker's body draws) is enabled exactly where the site
//  exists. A topology change made through the model mutators the controls
//  bind to leaves an appeared site asking for a choice and a disappeared one
//  holding `does_not_apply`.
//
//  The drawn controls themselves cannot be read back here: SwiftUI builds no
//  accessibility tree for an in-process query (`accessibilityChildren()` of
//  the hosting view is empty without an accessibility client) and draws these
//  pickers as graphics views, not `NSPopUpButton`s. That every picker is
//  present on screen, the fusion picker included with the feature skip off,
//  is checked on the running app (plan V7).
//

import AppKit
import SwiftUI
import XCTest
@testable import DrewsChessMachine

@MainActor
final class BuildNewModelSiteActivationRenderTests: XCTestCase {

    /// Lets SwiftUI finish the update a model change scheduled.
    private func settle(_ host: NSView) {
        host.layoutSubtreeIfNeeded()
        RunLoop.main.run(until: Date().addingTimeInterval(0.3))
        host.layoutSubtreeIfNeeded()
    }

    /// Hosts the screen for `model`; the window is closed at teardown.
    private func hostScreen(_ model: BuildNewModelModel) -> NSView {
        let view = BuildNewModelView(model: model, onBuild: { _ in }, onCancel: {})
        let host = NSHostingView(rootView: view.frame(width: 1000, height: 8000))
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1000, height: 8000),
            styleMask: [.titled],
            backing: .buffered,
            defer: false
        )
        window.isReleasedWhenClosed = false
        window.contentView = host
        window.orderFrontRegardless()
        addTeardownBlock { @MainActor in
            window.orderOut(nil)
            window.contentView = nil
        }
        settle(host)
        return host
    }

    private func presentation(_ model: BuildNewModelModel, _ site: ArchitectureActivationSite)
        -> ArchitectureSiteActivationPicker.Presentation {
        ArchitectureSiteActivationPicker.presentation(
            site: site, activation: model.storedActivation(at: site), siteExists: model.siteExists(site))
    }

    /// Every site's picker presentation is enabled exactly where the site
    /// exists, and offers the functions (a site in a consistent state needs no
    /// choice).
    private func assertPickersMatchTheTopology(_ model: BuildNewModelModel, _ context: String) {
        let functions = ActivationFunction.functions.map {
            ArchitectureSiteActivationPicker.Entry(value: $0, title: $0.rawValue)
        }
        for site in ArchitectureActivationSite.allCases {
            let shown = presentation(model, site)
            XCTAssertEqual(shown.isEnabled, model.siteExists(site), "\(context): \(site.displayName)")
            if model.siteExists(site) {
                XCTAssertFalse(shown.needsChoice, "\(context): \(site.displayName)")
                XCTAssertEqual(shown.entries, functions, "\(context): \(site.displayName)")
            } else {
                XCTAssertEqual(model.storedActivation(at: site), .doesNotApply, "\(context): \(site.displayName)")
                XCTAssertEqual(shown.entries, [.init(value: .doesNotApply, title: "does not apply")],
                               "\(context): \(site.displayName)")
            }
        }
    }

    private func makeModel(_ arch: NetworkArchitecture) -> BuildNewModelModel {
        BuildNewModelModel(NamedArchitecture(label: "test", architecture: arch))
    }

    func testEverySitePickerIsEnabledExactlyWhereItsSiteExists() throws {
        var postToPost = ArchitectureActivationSiteTests.tiny(style: .post)
        postToPost.blockGroups.append(postToPost.blockGroups[0])
        var preToPost = ArchitectureActivationSiteTests.tiny()
        preToPost.blockGroups.append(ArchitectureActivationSiteTests.tiny(style: .post).blockGroups[0])
        preToPost.clearActivationSitesTheTopologyLacks()
        let configurations: [(String, NetworkArchitecture)] = [
            ("pre → pre, intermediate_conv, no skip", ArchitectureActivationSiteTests.tiny()),
            ("post → post, intermediate_conv", postToPost),
            ("pre → post", preToPost),
            ("simple_conv", ArchitectureActivationSiteTests.tiny(policy: .simpleConv)),
            ("post → pre, compress fusion", ArchitectureActivationSiteTests.fullSiteFixture()),
        ]
        for (name, arch) in configurations {
            try arch.validate()
            let model = makeModel(arch)
            _ = hostScreen(model)
            XCTAssertTrue(model.isValid, "\(name): \(model.validationError ?? "")")
            assertPickersMatchTheTopology(model, name)
        }
    }

    func testATopologyChangeShowsAChoiceForAnAppearedSite() throws {
        let model = makeModel(ArchitectureActivationSiteTests.tiny(policy: .simpleConv))
        let host = hostScreen(model)
        assertPickersMatchTheTopology(model, "simple_conv")

        model.policyHeadStyle = .intermediateConv
        model.blockGroupDrafts[0].activationStyle = .post
        model.featureSkipSource = .stemOutput
        model.featureSkipFusion = .compressConvBNReLU
        model.featureSkipToValueHead = true
        settle(host)
        for site in [ArchitectureActivationSite.stem, .featureSkipFusion, .policyHead] {
            XCTAssertTrue(model.siteExists(site), "\(site)")
            XCTAssertEqual(model.storedActivation(at: site), .doesNotApply, "\(site): an appeared site asks for a choice")
            let shown = presentation(model, site)
            XCTAssertTrue(shown.isEnabled, "\(site)")
            XCTAssertTrue(shown.needsChoice, "\(site)")
            XCTAssertEqual(shown.entries.first, .init(value: .doesNotApply, title: "choose…"), "\(site)")
            XCTAssertEqual(Array(shown.entries.dropFirst()).map(\.value), ActivationFunction.functions, "\(site)")
        }
        XCTAssertFalse(model.siteExists(.towerEnd), "the post-activation tower lost its tower end")
        XCTAssertEqual(model.towerEndActivation, .doesNotApply)
        XCTAssertFalse(presentation(model, .towerEnd).isEnabled)
        XCTAssertNil(model.buildRequest)

        model.stemActivation = .relu
        model.featureSkipActivation = .relu
        model.policyHeadActivation = .leakyRelu
        settle(host)
        XCTAssertTrue(model.isValid, model.validationError ?? "")
        assertPickersMatchTheTopology(model, "after choosing")
    }
}
