import SwiftUI

/// App ▸ About: the standard panel plus the build counter and commit
/// (`AboutPanel`). Its own `Commands` type so the app's `body` stays small
/// enough to type-check quickly.
struct AboutCommands: Commands {
    var body: some Commands {
        CommandGroup(replacing: .appInfo) {
            Button("About \(AppVersionText.appName)") { AboutPanel.show() }
        }
    }
}
