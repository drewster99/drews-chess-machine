//
//  RelativeGradientCapSection.swift
//  DrewsChessMachine
//
//  The relative gradient cap's settings on the settings popover's Optimizer
//  tab, directly under the "Clip:" row (the hard max it tightens): a
//  segmented Off / Log only / Clip picker and the k, N, W and floor rows.
//  The rows stay in the hierarchy whatever the mode — dimmed and disabled
//  when Off — so the popover keeps its height and a stored value that is not
//  in effect stays visibly present-but-inactive (the policy-smoothing rows'
//  treatment). Values are transactional: the parent's Save validates them
//  (including W ≤ N) and writes them.
//

import SwiftUI

struct RelativeGradientCapSection: View {
    @Binding var mode: RelativeGradientCapMode
    @Binding var multipleText: String
    @Binding var windowStepsText: String
    @Binding var minimumHistoryStepsText: String
    @Binding var floorText: String
    let multipleError: Bool
    let windowStepsError: Bool
    let minimumHistoryStepsError: Bool
    let floorError: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            RelativeGradientCapModePicker(mode: $mode)
            RelativeGradientCapValueRows(
                inactive: mode == .off,
                multipleText: $multipleText, windowStepsText: $windowStepsText,
                minimumHistoryStepsText: $minimumHistoryStepsText, floorText: $floorText,
                multipleError: multipleError, windowStepsError: windowStepsError,
                minimumHistoryStepsError: minimumHistoryStepsError, floorError: floorError
            )
        }
    }
}

private struct RelativeGradientCapModePicker: View {
    @Binding var mode: RelativeGradientCapMode

    var body: some View {
        HStack(spacing: 8) {
            Text("Relative clip:")
                .frame(width: 160, alignment: .trailing)
            Picker("", selection: $mode) {
                ForEach(RelativeGradientCapMode.allCases, id: \.self) { mode in
                    Text(mode.displayName).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()
            Spacer()
        }
    }
}

private struct RelativeGradientCapValueRows: View {
    let inactive: Bool
    @Binding var multipleText: String
    @Binding var windowStepsText: String
    @Binding var minimumHistoryStepsText: String
    @Binding var floorText: String
    let multipleError: Bool
    let windowStepsError: Bool
    let minimumHistoryStepsError: Bool
    let floorError: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            PopoverRow(label: "k × median:", text: $multipleText, error: multipleError,
                       placeholder: RelativeGradClipMultiple.declaredDefaultText(format: "%.1f"),
                       hint: "of pre-clip norms", disabled: inactive) {
                Stepper("", value: PopoverBindings.doubleBinding(
                    text: $multipleText, fallback: RelativeGradClipMultiple.declaredDefault, format: "%.1f"),
                        in: RelativeGradClipMultiple.declaredClosedRange, step: 0.5)
            }
            RelativeGradientCapStepRows(
                inactive: inactive, windowStepsText: $windowStepsText,
                minimumHistoryStepsText: $minimumHistoryStepsText,
                windowStepsError: windowStepsError, minimumHistoryStepsError: minimumHistoryStepsError)
            PopoverRow(label: "Floor:", text: $floorText, error: floorError,
                       placeholder: RelativeGradClipFloor.declaredDefaultText(format: "%.2f"),
                       hint: "lowest relative cap", disabled: inactive) {
                Stepper("", value: PopoverBindings.doubleBinding(
                    text: $floorText, fallback: RelativeGradClipFloor.declaredDefault, format: "%.2f"),
                        in: RelativeGradClipFloor.declaredClosedRange, step: 0.1)
            }
        }
        .opacity(inactive ? 0.5 : 1)
        .disabled(inactive)
    }
}

private struct RelativeGradientCapStepRows: View {
    let inactive: Bool
    @Binding var windowStepsText: String
    @Binding var minimumHistoryStepsText: String
    let windowStepsError: Bool
    let minimumHistoryStepsError: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            PopoverRow(label: "Window N:", text: $windowStepsText, error: windowStepsError,
                       placeholder: String(RelativeGradClipWindowSteps.declaredDefault),
                       hint: "steps in the median", disabled: inactive) {
                Stepper("", value: PopoverBindings.intBinding(
                    text: $windowStepsText, fallback: RelativeGradClipWindowSteps.declaredDefault),
                        in: RelativeGradClipWindowSteps.declaredClosedRange, step: 100)
            }
            PopoverRow(label: "Min history W:", text: $minimumHistoryStepsText, error: minimumHistoryStepsError,
                       placeholder: String(RelativeGradClipMinHistorySteps.declaredDefault),
                       hint: "warm-up steps, ≤ N", disabled: inactive) {
                Stepper("", value: PopoverBindings.intBinding(
                    text: $minimumHistoryStepsText, fallback: RelativeGradClipMinHistorySteps.declaredDefault),
                        in: RelativeGradClipMinHistorySteps.declaredClosedRange, step: 10)
            }
        }
    }
}
