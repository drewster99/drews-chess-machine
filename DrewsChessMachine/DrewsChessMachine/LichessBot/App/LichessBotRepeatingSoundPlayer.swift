import AppKit
import Foundation

/// Plays system sounds a set number of times each, one play after another,
/// and queued requests after the one playing.
///
/// Why plays are sequenced by the sound's own "finished" callback
/// (`NSSoundDelegate`) rather than by sleeping for the sound's duration or
/// polling `isPlaying`: the callback is exact (no gap, no overlap, whatever
/// the device's output latency), costs nothing while a sound plays, and
/// never blocks the main actor. Each request plays a private copy of the
/// sound (`LichessBotSystemSounds.makeSound(named:)`), so this player's
/// delegate never collides with another use of the shared named sound.
///
/// Requests are queued, not mixed: two human games starting together
/// sound one alert after the other, each still recognizable.
///
/// A play that fails to start, or ends early, is reported to the request's
/// `onFailure` (never dropped silently) and that request's remaining plays
/// are abandoned; the queue moves on.
@MainActor
final class LichessBotRepeatingSoundPlayer: NSObject, NSSoundDelegate {
    enum Failure: LocalizedError {
        case repeatCountOutOfRange(Int)
        case playbackDidNotStart(soundName: String)
        case playbackEndedEarly(soundName: String)

        var errorDescription: String? {
            switch self {
            case .repeatCountOutOfRange(let count):
                let range = LichessBotLimits.alertSoundRepeatCountRange
                return "A sound can be repeated \(range.lowerBound) to \(range.upperBound) times, not \(count)"
            case .playbackDidNotStart(let soundName):
                return "System sound \"\(soundName)\" did not start playing"
            case .playbackEndedEarly(let soundName):
                return "System sound \"\(soundName)\" stopped before it finished"
            }
        }
    }

    private struct Request {
        let soundName: String
        let sound: NSSound
        /// Plays still to start, counting the one playing.
        var playsLeft: Int
        let onFailure: @MainActor (any Error) -> Void
    }

    /// Requests waiting their turn, oldest first.
    private var queue: [Request] = []
    /// The request whose sound is playing now.
    private var playing: Request?

    /// Queue `soundName` to play `times` times in a row. Throws at once for
    /// a repeat count outside `LichessBotLimits.alertSoundRepeatCountRange`
    /// or a sound that doesn't exist; a failure while playing goes to
    /// `onFailure`.
    func enqueue(soundName: String, times: Int, onFailure: @escaping @MainActor (any Error) -> Void) throws {
        guard LichessBotLimits.alertSoundRepeatCountRange.contains(times) else {
            throw Failure.repeatCountOutOfRange(times)
        }
        let sound = try LichessBotSystemSounds.makeSound(named: soundName)
        sound.delegate = self
        queue.append(Request(soundName: soundName, sound: sound, playsLeft: times, onFailure: onFailure))
        if playing == nil {
            playNextRequest()
        }
    }

    /// Stop the playing sound and drop every queued request (a preview
    /// replaced by a newer one). The stopped sound's delegate is cleared
    /// first, so its stop is not reported as a failure.
    func stopAll() {
        queue.removeAll()
        guard let current = playing else { return }
        playing = nil
        current.sound.delegate = nil
        current.sound.stop()
    }

    private func playNextRequest() {
        guard playing == nil, !queue.isEmpty else { return }
        start(queue.removeFirst())
    }

    private func start(_ request: Request) {
        if request.sound.play() {
            playing = request
        } else {
            request.sound.delegate = nil
            request.onFailure(Failure.playbackDidNotStart(soundName: request.soundName))
            playNextRequest()
        }
    }

    func sound(_ sound: NSSound, didFinishPlaying flag: Bool) {
        guard var current = playing, current.sound === sound else { return }
        playing = nil
        guard flag else {
            current.sound.delegate = nil
            current.onFailure(Failure.playbackEndedEarly(soundName: current.soundName))
            playNextRequest()
            return
        }
        current.playsLeft -= 1
        if current.playsLeft > 0 {
            start(current)
        } else {
            current.sound.delegate = nil
            playNextRequest()
        }
    }
}
