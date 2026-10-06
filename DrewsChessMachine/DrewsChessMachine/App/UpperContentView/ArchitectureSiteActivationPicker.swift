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
/// Whether the site exists, and the help for that state, is one
/// `Availability`.
struct ArchitectureSiteActivationPicker: View {
    let title: String
    @Binding var activation: ActivationFunction
    let availability: Availability

    /// Whether the site exists, with the help text for that state. Each
    /// text is read only in the state it describes.
    enum Availability {
        /// The site exists; the help says what its activation is applied to.
        case exists(siteDescription: String)
        /// The topology lacks the site; the help says why.
        case absent(reason: String)

        /// An architecture-level site.
        static func of(_ site: ArchitectureActivationSite, siteExists: Bool) -> Availability {
            siteExists ? .exists(siteDescription: site.siteDescription) : .absent(reason: site.absentReason)
        }

        /// A block group's SE FC1 (`BlockGroup.hasSEFC1`): the screen's SE
        /// picker and its test read this one rule.
        static func seFC1(of group: BlockGroup) -> Availability {
            group.hasSEFC1
                ? .exists(siteDescription: "Activation after the SE bottleneck FC1 (C → C/r), set independently "
                    + "of the group's activation. leaky_relu here keeps FC1 units from dying at almost no cost.")
                : .absent(reason: BlockGroup.seLessReason)
        }
    }

    /// An architecture-level site's picker on the Build screen: binds the
    /// model's field for `site` (`BuildNewModelModel.activationKeyPath(at:)`)
    /// and is enabled when `existingSites` holds `site`.
    init(site: ArchitectureActivationSite, model: BuildNewModelModel,
         existingSites: Set<ArchitectureActivationSite>) {
        self.init(title: site.displayName,
                  activation: Bindable(model)[dynamicMember: BuildNewModelModel.activationKeyPath(at: site)],
                  availability: .of(site, siteExists: existingSites.contains(site)))
    }

    /// Any topology-dependent site's picker (a block group's SE FC1 uses it).
    init(title: String, activation: Binding<ActivationFunction>, availability: Availability) {
        self.title = title
        _activation = activation
        self.availability = availability
    }

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

    static func presentation(activation: ActivationFunction, availability: Availability) -> Presentation {
        switch availability {
        case .absent(let reason):
            return Presentation(
                isEnabled: false, needsChoice: false,
                entries: [Entry(value: .doesNotApply, title: "does not apply")],
                help: "Does not apply: \(reason).")
        case .exists(let siteDescription):
            let needsChoice = activation == .doesNotApply
            let choose = needsChoice ? [Entry(value: .doesNotApply, title: "choose…")] : []
            return Presentation(
                isEnabled: true, needsChoice: needsChoice,
                entries: choose + ActivationFunction.functions.map { Entry(value: $0, title: $0.rawValue) },
                help: siteDescription)
        }
    }

    /// What this picker draws for its current state.
    var presentation: Presentation {
        Self.presentation(activation: activation, availability: availability)
    }

    var body: some View {
        let presentation = self.presentation
        Picker(
            selection: $activation,
            content: {
                ForEach(presentation.entries, id: \.value) { entry in
                    Text(entry.title).tag(entry.value)
                }
            },
            label: {
                Text(title)
                    .foregroundStyle(presentation.needsChoice ? Color.orange : Color.primary)
            }
        )
        .disabled(!presentation.isEnabled)
        .help(presentation.help)
    }
}
