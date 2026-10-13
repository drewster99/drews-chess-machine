import AppKit
import Foundation

/// The macOS system alert sounds, read from the folder they live in rather
/// than listed by hand, so the choices always match what `NSSound(named:)`
/// can find on this machine.
enum LichessBotSystemSounds {
    enum Failure: LocalizedError {
        case soundNotFound(String)
        case soundNotCopyable(String)

        var errorDescription: String? {
            switch self {
            case .soundNotFound(let name):
                return "System sound \"\(name)\" was not found"
            case .soundNotCopyable(let name):
                return "System sound \"\(name)\" could not be copied for playback"
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

    /// A private copy of the named system sound. `NSSound(named:)` hands
    /// every caller the same shared instance, which plays only once at a time
    /// and has one delegate; a copy can be given its own delegate (to learn
    /// when it finishes) without taking over, or being cut off by, any other
    /// use of that sound.
    @MainActor
    static func makeSound(named name: String) throws -> NSSound {
        guard let shared = NSSound(named: NSSound.Name(name)) else {
            throw Failure.soundNotFound(name)
        }
        guard let copy = shared.copy() as? NSSound else {
            throw Failure.soundNotCopyable(name)
        }
        return copy
    }
}
