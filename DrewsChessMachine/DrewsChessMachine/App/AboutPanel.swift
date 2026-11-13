//
//  AboutPanel.swift
//
//  App ▸ About. The standard panel shows only "Version <marketing> (<bundle
//  version>)", and the bundle version doesn't move from build to build, so
//  two different builds look identical there. This keeps the standard panel
//  (icon, name, version line) and adds the build counter, commit, date and
//  configuration in its credits area, centered under the version.
//

import AppKit

@MainActor
enum AboutPanel {

    static func show() {
        let paragraph = NSMutableParagraphStyle()
        paragraph.alignment = .center
        let credits = NSAttributedString(
            string: AppVersionText.buildLine,
            attributes: [
                .font: NSFont.monospacedDigitSystemFont(ofSize: NSFont.smallSystemFontSize, weight: .regular),
                .foregroundColor: NSColor.secondaryLabelColor,
                .paragraphStyle: paragraph
            ]
        )
        NSApplication.shared.activate()
        NSApplication.shared.orderFrontStandardAboutPanel(options: [.credits: credits])
    }
}
