import SwiftUI

/// One of DCM's challenges waiting for an answer, on the Overview: the
/// player (a link to their breakdown window), when it was sent, what the
/// player is doing now (`LichessBotPendingOpponentStatuses`), and Cancel.
struct LichessBotPendingChallengeRow: View {
    let controller: LichessBotController
    let pending: LichessBotController.PendingChallenge

    /// A player in another game is the usual reason a bot leaves a
    /// challenge unanswered; it stands out from the rest.
    static let playingColor: Color = .orange
    static let otherActivityColor: Color = .secondary

    var body: some View {
        let activity = controller.pendingOpponentStatuses.activity(of: pending.username)
        HStack(spacing: 8) {
            LichessBotFavoriteStar(controller: controller, userID: pending.username)
            Text("Challenge to")
                .font(.callout)
            Button(pending.username) {
                LichessBotOpponentWindowLauncher.open(opponentID: pending.username, name: pending.username, controller: controller)
            }
            .buttonStyle(.link)
            .font(.callout)
            .help("Open this opponent's breakdown")
            Text("waiting since \(pending.sentAt.formatted(date: .omitted, time: .standard))")
                .font(.callout)
            Text(Self.activityText(activity))
                .font(.callout)
                .foregroundStyle(Self.isPlaying(activity) ? Self.playingColor : Self.otherActivityColor)
                .help(Self.activityHelp(activity))
            Button("Cancel") {
                Task { await controller.cancelChallenge(id: pending.id) }
            }
        }
    }

    static func isPlaying(_ activity: LichessBotPendingOpponentStatuses.Activity) -> Bool {
        if case .playing = activity { return true }
        return false
    }

    static func activityText(_ activity: LichessBotPendingOpponentStatuses.Activity) -> String {
        switch activity {
        case .playing:
            return "· playing a game"
        case .onlineNotPlaying:
            return "· online, not playing"
        case .offline:
            return "· offline"
        case .notReported:
            return "· no Lichess status"
        case .notReadYet:
            return "· checking…"
        case .unknown:
            return "· status unknown"
        }
    }

    static func activityHelp(_ activity: LichessBotPendingOpponentStatuses.Activity) -> String {
        switch activity {
        case .playing(let readAt), .onlineNotPlaying(let readAt), .offline(let readAt), .notReported(let readAt):
            return "Lichess status as of \(readAt.formatted(date: .omitted, time: .standard)); read every \(Int(LichessBotPendingOpponentStatuses.readInterval)) s while challenges are pending"
        case .notReadYet:
            return "Lichess status not read yet"
        case .unknown(let failure):
            return "Reading the Lichess status failed: \(failure)"
        }
    }
}
