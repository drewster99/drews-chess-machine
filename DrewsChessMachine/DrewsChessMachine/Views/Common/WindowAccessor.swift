import AppKit
import SwiftUI

/// Captures the hosting `NSWindow` of a SwiftUI view into a
/// `@State` binding. Needed because the global
/// `NSWindow.willCloseNotification` fires for every NSWindow the
/// app ever puts on screen — including the Log Analysis auxiliary
/// window (`LogAnalysisWindowController`) and any NSOpenPanel /
/// NSSavePanel the user raises via the File menu. Comparing the
/// notification's `object` against this captured pointer lets the
/// teardown hook ignore anything that isn't the main ContentView
/// window.
///
/// The dispatch to main is required: `NSView.window` is nil during
/// `makeNSView` because the view hasn't been inserted into a
/// window yet. Deferring by one runloop tick lets the parent
/// finish attaching and gives us a valid window reference.
///
/// `onAttached` (optional): fires on the same runloop tick as the
/// window-pointer assignment. Unlike `.onAppear`, this hook is NOT
/// gated on the window becoming key / front / visible — it runs as
/// soon as the SwiftUI view tree is materialized into AppKit. That
/// makes it the right place to trigger startup work (e.g. the
/// `--train` headless launch sequence) that doesn't actually need
/// the window to be visible. Used together with `onAppear` for
/// concerns that DO need the window to be visible / focused, this
/// gives us two distinct lifecycle gates: "view tree is ready" vs
/// "user-visible window is up."
struct WindowAccessor: NSViewRepresentable {
    @Binding var window: NSWindow?
    var onAttached: (() -> Void)? = nil
    func makeNSView(context: Context) -> NSView {
        let v = NSView()
        DispatchQueue.main.async {
            self.window = v.window
            self.onAttached?()
        }
        return v
    }
    func updateNSView(_ nsView: NSView, context: Context) {}
}
