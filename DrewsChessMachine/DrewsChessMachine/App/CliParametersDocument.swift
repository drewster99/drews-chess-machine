import SwiftUI
import UniformTypeIdentifiers

/// `FileDocument` adapter for the `Save Parameters…` file exporter.
/// Holds the already-encoded parameters JSON as raw `Data` so the
/// callsite can build the JSON on the main actor (where it has access
/// to all the `@AppStorage` / `@State` fields) and hand the bytes off
/// to SwiftUI's exporter machinery without doing any further work.
/// Read support is unused — the import path uses `.fileImporter`,
/// which doesn't require a `FileDocument` — but we conform anyway
/// since `FileDocument` requires both initializers.
struct CliParametersDocument: FileDocument {
    static let readableContentTypes: [UTType] = [.json]
    static let writableContentTypes: [UTType] = [.json]

    var data: Data

    init(data: Data) { self.data = data }

    init(configuration: ReadConfiguration) throws {
        guard let bytes = configuration.file.regularFileContents else {
            throw CocoaError(.fileReadCorruptFile)
        }
        self.data = bytes
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}
