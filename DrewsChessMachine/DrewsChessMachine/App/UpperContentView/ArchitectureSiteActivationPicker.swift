//
//  ArchitectureSiteActivationPicker.swift
//  DrewsChessMachine
//
//  The Build-New-Model screen's picker for one architecture-level activation
//  site (stem, tower end, fusion, policy pre-block, value conv, value FC1
//  hidden).
//

import SwiftUI

/// One architecture-level activation site's picker. Always present, so the
/// form's rows never shift as the topology changes (the SE-activation
/// precedent):
/// - a site the topology lacks is disabled and shows "does not apply", with
///   the reason as help;
/// - a site that exists lists `ActivationFunction.functions`, plus a
///   "choose…" entry only while it still holds `does_not_apply` (a site that
///   has just appeared). Its label turns orange then, and the screen's
///   validation message names the site until a function is chosen. The
///   earlier choice is never restored on its own.
struct ArchitectureSiteActivationPicker: View {
    let site: ArchitectureActivationSite
    @Binding var activation: ActivationFunction
    let siteExists: Bool

    /// The functions an existing site offers: `ActivationFunction.functions`,
    /// never `allCases` (which includes the `does_not_apply` marker).
    static let functionChoices = ActivationFunction.functions

    /// One entry of the picker's menu.
    struct Entry: Equatable {
        let value: ActivationFunction
        let title: String
    }

    /// What the picker shows for a site in a given state — the one rule the
    /// body draws, kept as a value so it can be checked without reading the
    /// drawn controls (SwiftUI exposes no accessibility tree to an in-process
    /// query, and draws these pickers as graphics views).
    struct Presentation: Equatable {
        let isEnabled: Bool
        /// The label is drawn in the screen's invalid colour: the site exists
        /// but no function has been chosen yet.
        let needsChoice: Bool
        let entries: [Entry]
        let help: String
    }

    static func presentation(
        site: ArchitectureActivationSite, activation: ActivationFunction, siteExists: Bool
    ) -> Presentation {
        guard siteExists else {
            return Presentation(
                isEnabled: false, needsChoice: false,
                entries: [Entry(value: .doesNotApply, title: "does not apply")],
                help: "Does not apply: \(site.absentReason).")
        }
        let needsChoice = activation == .doesNotApply
        let choose = needsChoice ? [Entry(value: .doesNotApply, title: "choose…")] : []
        return Presentation(
            isEnabled: true, needsChoice: needsChoice,
            entries: choose + functionChoices.map { Entry(value: $0, title: $0.rawValue) },
            help: site.siteDescription)
    }

    var body: some View {
        let presentation = Self.presentation(site: site, activation: activation, siteExists: siteExists)
        Picker(selection: $activation) {
            ForEach(presentation.entries, id: \.value) { entry in
                Text(entry.title).tag(entry.value)
            }
        } label: {
            Text(site.displayName)
                .foregroundStyle(presentation.needsChoice ? Color.orange : Color.primary)
        }
        .disabled(!presentation.isEnabled)
        .help(presentation.help)
    }
}
