import Foundation

/// Writes the default-parameters files for `--create-parameters-file`:
/// `parameters.json` (every parameter's default) and `parameters.md` (the
/// categorized description).
///
/// The path names either a folder or the JSON file:
/// - an existing folder, or any path ending in `/`, gets `parameters.json`
///   and `parameters.md` inside it (the folder must exist);
/// - anything else is the JSON file itself, with the markdown written beside
///   it under the same name and a `.md` extension.
///
/// `--force` replaces existing *regular files* only. Anything else at a
/// destination — a folder, a symbolic link — is refused with or without
/// `--force`. Treating a folder path as the file name once made `--force`
/// delete a whole documentation folder recursively to put the JSON file in
/// its place; a destination is therefore never removed, only atomically
/// replaced when it is a regular file.
enum ParametersFileWriter {

    static let jsonFileName = "parameters.json"
    static let markdownFileName = "parameters.md"

    /// Where `path` puts the two files. Throws when `path` ends in `/` but
    /// no such folder exists.
    static func destinations(forPath path: String) throws -> (json: URL, markdown: URL) {
        let expanded = (path as NSString).expandingTildeInPath
        var isFolder: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: expanded, isDirectory: &isFolder)
        // Read the trailing `/` from `path` itself: tilde expansion drops it.
        if exists && isFolder.boolValue || path.hasSuffix("/") {
            guard exists && isFolder.boolValue else {
                throw ParametersFileWriterError.folderMissing(expanded)
            }
            let folder = URL(fileURLWithPath: expanded, isDirectory: true)
            return (folder.appendingPathComponent(jsonFileName), folder.appendingPathComponent(markdownFileName))
        }
        let jsonURL = URL(fileURLWithPath: expanded)
        return (jsonURL, jsonURL.deletingPathExtension().appendingPathExtension("md"))
    }

    /// Write both files for `path` and return where they went. `force`
    /// allows replacing an existing `parameters.json`; the markdown beside it
    /// is replaced whenever the JSON is written. Nothing is written when the
    /// request is refused.
    static func writeDefaults(path: String, force: Bool) throws -> (json: URL, markdown: URL) {
        let (jsonURL, mdURL) = try destinations(forPath: path)
        let jsonExists = try requireRegularFileIfPresent(jsonURL)
        _ = try requireRegularFileIfPresent(mdURL)
        if jsonExists && !force {
            throw ParametersFileWriterError.destinationExists(jsonURL.path)
        }

        let jsonData = try TrainingParameters.defaultsJSON()
        let mdData = Data(TrainingParameters.defaultsMarkdown().utf8)
        try writeReplacing(jsonData, at: jsonURL)
        try writeReplacing(mdData, at: mdURL)
        return (jsonURL, mdURL)
    }

    /// Whether `url` exists; throws if it exists and is not a regular file.
    /// Reads the item itself, not a symbolic link's target.
    private static func requireRegularFileIfPresent(_ url: URL) throws -> Bool {
        let attributes: [FileAttributeKey: Any]
        do {
            attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        } catch let error as CocoaError where error.code == .fileReadNoSuchFile || error.code == .fileNoSuchFile {
            return false
        }
        guard let type = attributes[.type] as? FileAttributeType else {
            throw ParametersFileWriterError.destinationNotAFile(url.path, kind: "unknown")
        }
        guard type == .typeRegular else {
            throw ParametersFileWriterError.destinationNotAFile(url.path, kind: type.rawValue)
        }
        return true
    }

    /// Write `data` to `url` atomically: a regular file already there is
    /// swapped for the new contents in one step, never removed first.
    private static func writeReplacing(_ data: Data, at url: URL) throws {
        // `.atomic` writes a temporary file in the same folder and renames it
        // over `url`; the rename replaces a regular file, and the checks above
        // guarantee nothing else is there.
        try data.write(to: url, options: [.atomic])
    }
}

enum ParametersFileWriterError: LocalizedError, Equatable {
    case destinationExists(String)
    case destinationNotAFile(String, kind: String)
    case folderMissing(String)

    var errorDescription: String? {
        switch self {
        case .destinationExists(let path):
            return "\(path) already exists; pass --force to overwrite"
        case .destinationNotAFile(let path, let kind):
            return "\(path) exists and is not a regular file (\(kind)); refusing to replace it, even with --force"
        case .folderMissing(let path):
            return "\(path) names a folder that does not exist"
        }
    }
}
