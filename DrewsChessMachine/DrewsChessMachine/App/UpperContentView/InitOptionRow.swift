//
//  InitOptionRow.swift
//  DrewsChessMachine
//
//  One init-neutral option's control on the Build-New-Model screen, marked
//  when its value differs from the standard init.
//

import SwiftUI

/// Wraps one init-neutral option's control (determinism plan B2.1). When the
/// value differs from the standard init the row gets an accent tint AND a
/// marker glyph — never color alone — and its tooltip says what the value
/// changes at step 0. The comparison is always against the standard value
/// (the caller passes `isNonStandard` from
/// `NetworkArchitecture.nonStandardInitOptions`), never against the last
/// edit, so a reopened model or preset still shows what is non-standard.
///
/// The glyph and the tint are always present and only fade, so marking or
/// clearing a row never moves the controls around it.
struct InitOptionRow<Content: View>: View {
    let isNonStandard: Bool
    let stepZeroEffect: String
    let content: Content

    init(isNonStandard: Bool, stepZeroEffect: String, @ViewBuilder content: () -> Content) {
        self.isNonStandard = isNonStandard
        self.stepZeroEffect = stepZeroEffect
        self.content = content()
    }

    var body: some View {
        HStack(spacing: 6) {
            content
            Image(systemName: "diamond.fill")
                .imageScale(.small)
                .foregroundStyle(Color.accentColor)
                .opacity(isNonStandard ? 1 : 0)
                .accessibilityLabel("Differs from the standard init")
                .accessibilityHidden(!isNonStandard)
        }
        .padding(.horizontal, 4)
        .padding(.vertical, 1)
        .background(
            RoundedRectangle(cornerRadius: 4)
                .fill(Color.accentColor.opacity(isNonStandard ? 0.12 : 0))
        )
        .help(isNonStandard ? "Differs from the standard init. \(stepZeroEffect)" : stepZeroEffect)
    }
}
