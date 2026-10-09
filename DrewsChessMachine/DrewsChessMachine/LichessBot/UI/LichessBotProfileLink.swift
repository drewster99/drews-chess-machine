import SwiftUI

/// The bot's own Lichess profile in the controls card: the address opens the
/// page in the browser, and the button beside it copies the address. The
/// account the token was checked against wins; before any check, the account
/// the settings expect (the same name the Account card shows).
struct LichessBotProfileLink: View {
    let controller: LichessBotController

    var body: some View {
        HStack(spacing: 4) {
            Button(displayedAddress) {
                LichessBotLinks.openUser(username)
            }
            .buttonStyle(.link)
            .font(.callout)
            .help("Open \(username)'s Lichess page in the browser")
            LichessBotCopyProfileLinkButton(username: username)
        }
    }

    private var username: String {
        controller.account?.username ?? controller.settings.connection.expectedAccountID
    }

    /// The address without its scheme, which the link style already implies.
    private var displayedAddress: String {
        String(LichessBotLinks.userAddress(username).trimmingPrefix("https://"))
    }
}
