import Foundation

struct TrainingAlarm: Identifiable, Equatable, Sendable {
    enum Severity: String, Sendable {
        case warning
        case critical
    }

    let id: UUID
    let severity: Severity
    let title: String
    let detail: String
    let raisedAt: Date
}
