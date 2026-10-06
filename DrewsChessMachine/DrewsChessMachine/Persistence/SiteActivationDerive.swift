//
//  SiteActivationDerive.swift
//  DrewsChessMachine
//
//  `--derive-model` operations for the architecture-level site activations
//  (format v9): one flag per `ArchitectureActivationSite` —
//  `--set-stem-activation`, `--set-tower-end-activation`,
//  `--set-feature-skip-activation`, `--set-policy-head-activation`,
//  `--set-value-head-conv-activation`, `--set-value-head-fc1-hidden-activation`.
//  Each sets only its own site's field, through
//  `NetworkArchitecture.setActivation(_:at:)`, the same checked setter every
//  other path uses. No activation has parameters, so no tensor is rewritten:
//  the derived file holds the source's weights bit-exact, which makes these
//  the A/B operations for "leaky value head on a ReLU tower" from one net,
//  and lets them run on a trained source too.
//
//  The catalog places them after `--set-activation`, so
//  `--set-activation X --set-value-head-fc1-hidden-activation Y` gives X
//  everywhere except Y. No derive operation changes which sites exist, so
//  none of them ever has to clear or choose a site.
//

import Foundation

/// Sets one architecture-level site's activation. Refuses `does_not_apply`
/// (it is not a function), a site the model's topology lacks (naming why it
/// is absent), an unknown value and a value the site already holds. The
/// source is untouched on every refusal.
struct SetSiteActivationDeriveOperation: DeriveOperation {
    let site: ArchitectureActivationSite
    let value: ActivationFunction

    /// The catalog entries, one per site in graph build order.
    static let kinds: [DeriveOperationKind] = ArchitectureActivationSite.allCases.map { kind(for: $0) }

    /// The operation name: `set-` and the site's JSON key in kebab case
    /// (`set-value-head-fc1-hidden-activation`).
    static func name(for site: ArchitectureActivationSite) -> String {
        "set-" + site.jsonKey.replacingOccurrences(of: "_", with: "-")
    }

    /// Why this operation refuses `does_not_apply`: the clause `doesNotApplyRefusal` appends.
    private static func doesNotApplyRefusalReason(for site: ArchitectureActivationSite) -> String {
        "--\(name(for: site)) sets only a site the model has"
    }

    static func kind(for site: ArchitectureActivationSite) -> DeriveOperationKind {
        let operationName = Self.name(for: site)
        return DeriveOperationKind(
            name: operationName,
            flag: "--\(operationName)",
            valueSyntax: ModelDerivation.activationFunctionValueSyntax,
            summary: "Set \(site.jsonKey) only. \(site.siteDescription) Refused on a model whose topology lacks "
                + "the site (it holds \(ActivationFunction.doesNotApply.rawValue) there). Applied after "
                + "--set-activation, so the two combine into \"this activation everywhere except here\". "
                + "Activations have no parameters, so every tensor is copied bit-exact.",
            changedArchitectureFields: [site.jsonKey],
            rewrittenTensorsDescription: "none",
            acceptsGroupSelection: false,
            make: { value, _ in
                SetSiteActivationDeriveOperation(
                    site: site,
                    value: try ModelDerivation.parseActivationFunctionValue(
                        value, operation: operationName, refusalReason: SetSiteActivationDeriveOperation.doesNotApplyRefusalReason(for: site)))
            })
    }

    var kindName: String { Self.name(for: site) }

    var recordedArguments: [String: String] { ["value": value.rawValue] }

    func apply(to architecture: NetworkArchitecture) throws -> NetworkArchitecture {
        guard value != .doesNotApply else {
            throw ModelDerivation.doesNotApplyRefusal(operation: kindName, refusalReason: Self.doesNotApplyRefusalReason(for: site))
        }
        guard architecture.hasActivationSite(site) else {
            throw ModelDerivation.DeriveError.operationNotApplicable(
                operation: kindName,
                detail: "the model has no \(site.jsonKey) site: \(site.absentReason)")
        }
        guard architecture.activation(at: site) != value else {
            throw ModelDerivation.DeriveError.operationNotApplicable(
                operation: kindName,
                detail: "\(site.jsonKey) is already '\(value.rawValue)'; nothing to derive")
        }
        var edited = architecture
        try edited.setActivation(value, at: site)
        return edited
    }

    func tensorRewrites(source: NetworkArchitecture, target: NetworkArchitecture) throws -> [DeriveTensorRewrite] {
        []
    }
}
