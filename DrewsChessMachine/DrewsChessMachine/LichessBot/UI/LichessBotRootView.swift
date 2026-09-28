import SwiftUI

/// The Lichess Bot window's content: a sidebar of sections (plan §14.3).
/// Every section stays mounted so its state — a browsed position, a
/// settings draft — survives switching away and back.
struct LichessBotRootView: View {
    let controller: LichessBotController
    @State private var section: Section = .overview
    @State private var showingFinishingSheet = false
    /// Held while the sheet is up, so it keeps its title while dismissing.
    @State private var finishingPurpose: LichessBotController.FinishingPurpose = .goOffline

    enum Section: String, CaseIterable, Identifiable {
        case overview = "Overview"
        case live = "Live Games"
        case settings = "Settings"

        var id: String { rawValue }

        var systemImage: String {
            switch self {
            case .overview: return "gauge.with.dots.needle.33percent"
            case .live: return "checkerboard.rectangle"
            case .settings: return "gearshape"
            }
        }
    }

    var body: some View {
        NavigationSplitView(
            sidebar: {
                List(Section.allCases, selection: $section) { section in
                    Label(section.rawValue, systemImage: section.systemImage)
                        .tag(section)
                }
                .navigationSplitViewColumnWidth(min: 160, ideal: 180, max: 220)
            },
            detail: {
                ZStack {
                    LichessBotOverviewView(controller: controller)
                        .shown(section == .overview)
                    LichessBotLiveView(controller: controller, isVisible: section == .live)
                        .shown(section == .live)
                    LichessBotSettingsView(controller: controller)
                        .shown(section == .settings)
                }
            }
        )
        .task {
            await controller.refreshIndex()
            await controller.loadPlayerNotes()
            if controller.tokenState == .unknown {
                await controller.refreshTokenState()
            }
        }
        // `initial`: the window may open with the sheet already due — a quit
        // or Go Offline opens it after setting `finishing`.
        .onChange(of: controller.finishing, initial: true) {
            Task { @MainActor in
                if let purpose = controller.finishing {
                    finishingPurpose = purpose
                }
                showingFinishingSheet = controller.finishing != nil
            }
        }
        .onChange(of: showingFinishingSheet) {
            Task { @MainActor in
                if !showingFinishingSheet && controller.finishing != nil {
                    controller.cancelFinishing()
                }
            }
        }
        .sheet(isPresented: $showingFinishingSheet) {
            LichessBotFinishingGamesSheet(controller: controller, purpose: finishingPurpose)
        }
    }
}
