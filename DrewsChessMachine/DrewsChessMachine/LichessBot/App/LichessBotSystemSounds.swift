import AppKit
import Foundation

/// The macOS system alert sounds, read from the folder they live in rather
/// than listed by hand, so the choices always match what `NSSound(named:)`
/// can find on this machine.
enum LichessBotSystemSounds {
    enum Failure: LocalizedError {
        case soundNotFound(String)

        var errorDescription: String? {
            switch self {
            case .soundNotFound(let name):
                return "System sound \"\(name)\" was not found"
            }
        }
    }

    static let directory = URL(fileURLWithPath: "/System/Library/Sounds", isDirectory: true)

    /// Sound names (file names without extension), sorted.
    static func availableNames() throws -> [String] {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        return files
            .map { $0.deletingPathExtension().lastPathComponent }
            .sorted()
    }

    /// Plays the named system sound. AppKit sound playback belongs on the
    /// main actor.
    @MainActor
    static func play(named name: String) throws {
        guard let sound = NSSound(named: NSSound.Name(name)) else {
            throw Failure.soundNotFound(name)
        }
        sound.play()
    }
}
