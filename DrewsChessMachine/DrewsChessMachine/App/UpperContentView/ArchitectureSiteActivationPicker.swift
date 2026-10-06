//
//  ArchitectureSiteActivationPicker.swift
//  DrewsChessMachine
//
//  The Build-New-Model screen's picker for one activation site that exists
//  only for some topologies: an architecture-level site (stem, tower end,
//  fusion, policy pre-block, value conv, value FC1 hidden) or a block group's
//  SE FC1.
//

import SwiftUI

/// One topology-dependent activation site's picker. Always present, so the
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
    let title: String
    @Binding var activation: ActivationFunction
    let siteExists: Bool
    /// Help for a site that exists: what the activation is applied to.
    let siteDescription: String
    /// Help for a site that does not: why it is absent.
    let absentReason: String

    /// An architecture-level site's picker.
    init(site: ArchitectureActivationSite, activation: Binding<ActivationFunction>, siteExists: Bool) {
        self.init(title: site.displayName, activation: activation, siteExists: siteExists,
                  siteDescription: site.siteDescription, absentReason: site.absentReason)
    }

    /// Any topology-dependent site's picker (a block group's SE FC1 uses it).
    init(title: String, activation: Binding<ActivationFunction>, siteExists: Bool,
         siteDescription: String, absentReason: String) {
        self.title = title
        _activation = activation
        self.siteExists = siteExists
        self.siteDescription = siteDescription
        self.absentReason = absentReason
    }

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
        presentation(activation: activation, siteExists: siteExists,
                     siteDescription: site.siteDescription, absentReason: site.absentReason)
    }

    static func presentation(
        activation: ActivationFunction, siteExists: Bool, siteDescription: String, absentReason: String
    ) -> Presentation {
        guard siteExists else {
            return Presentation(
                isEnabled: false, needsChoice: false,
                entries: [Entry(value: .doesNotApply, title: "does not apply")],
                help: "Does not apply: \(absentReason).")
        }
        let needsChoice = activation == .doesNotApply
        let choose = needsChoice ? [Entry(value: .doesNotApply, title: "choose…")] : []
        return Presentation(
            isEnabled: true, needsChoice: needsChoice,
            entries: choose + functionChoices.map { Entry(value: $0, title: $0.rawValue) },
            help: siteDescription)
    }

    var body: some View {
        let presentation = Self.presentation(
            activation: activation, siteExists: siteExists,
            siteDescription: siteDescription, absentReason: absentReason)
        Picker(selection: $activation) {
            ForEach(presentation.entries, id: \.value) { entry in
                Text(entry.title).tag(entry.value)
            }
        } label: {
            Text(title)
                .foregroundStyle(presentation.needsChoice ? Color.orange : Color.primary)
        }
        .disabled(!presentation.isEnabled)
        .help(presentation.help)
    }
}
