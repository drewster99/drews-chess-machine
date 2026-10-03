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
/// Either destination already existing needs `--force`, and `--force`
/// replaces existing *regular files* only. Anything else at a
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

    /// Write both files for `path` and return where they went. Either file
    /// already existing needs `force` — each destination is someone's file
    /// until proven otherwise (`--create-parameters-file ROADMAP` must not
    /// silently replace `ROADMAP.md`). Nothing is written when the request
    /// is refused.
    ///
    /// The two files are written one after the other, so the pair is
    /// all-or-nothing only as far as undoing the first write allows: when the
    /// markdown fails after the JSON was published as a new file, that JSON
    /// file — proven by identity to be the one just written — is removed
    /// again, so the failure leaves nothing behind. When the JSON replaced an
    /// existing file (`force`), its earlier contents are already gone and
    /// cannot be restored, so the new JSON stays and the error says so.
    static func writeDefaults(path: String, force: Bool) throws -> (json: URL, markdown: URL) {
        let (jsonURL, mdURL) = try destinations(forPath: path)
        guard try !FileSafety.mayNameTheSameFile(jsonURL, mdURL) else {
            throw ParametersFileWriterError.jsonAndMarkdownSamePath(jsonURL.path)
        }
        let existingJSON = try existingRegularFile(at: jsonURL)
        let existingMarkdown = try existingRegularFile(at: mdURL)
        if !force {
            if existingJSON != nil { throw ParametersFileWriterError.destinationExists(jsonURL.path) }
            if existingMarkdown != nil { throw ParametersFileWriterError.destinationExists(mdURL.path) }
        }

        let jsonData = try TrainingParameters.defaultsJSON()
        let mdData = Data(TrainingParameters.defaultsMarkdown().utf8)
        let writtenJSON = try write(jsonData, at: jsonURL, replacing: existingJSON)
        do {
            try write(mdData, at: mdURL, replacing: existingMarkdown)
        } catch {
            throw markdownFailure(error, markdownURL: mdURL, jsonURL: jsonURL,
                                  writtenJSON: writtenJSON, jsonReplacedAFile: existingJSON != nil)
        }
        return (jsonURL, mdURL)
    }

    /// The regular file at `url`, or nil when nothing is there; throws if
    /// something other than a regular file is. Reads the item itself, not a
    /// symbolic link's target.
    private static func existingRegularFile(at url: URL) throws -> FileSafety.ExistingItem? {
        guard let existing = try FileSafety.existingItem(at: url) else { return nil }
        guard existing.kind == .regularFile else {
            throw ParametersFileWriterError.destinationNotAFile(url.path, kind: existing.kind.description)
        }
        return existing
    }

    /// Write `data` to `url`, staged in a temporary sibling and renamed into
    /// place so the destination is never removed first and never left
    /// half-written, and return the written file's identity. With
    /// `replacing` nil (nothing was there when checked) the rename is
    /// exclusive, so a file that appeared since is refused rather than
    /// replaced; otherwise only that very regular file is replaced, so a
    /// different file, folder or link that took its place since is refused
    /// too.
    @discardableResult
    private static func write(_ data: Data, at url: URL, replacing existing: FileSafety.ExistingItem?) throws -> FileSafety.FileIdentity {
        do {
            if let existing {
                return try FileSafety.replaceRegularFile(data, at: url, expectedIdentity: existing.identity)
            }
            return try FileSafety.publishNewFile(data, to: url)
        } catch FileSafetyError.alreadyExists(path: _, kind: .regularFile) {
            throw ParametersFileWriterError.destinationExists(url.path)
        } catch FileSafetyError.alreadyExists(path: _, kind: let kind) {
            throw ParametersFileWriterError.destinationNotAFile(url.path, kind: kind.description)
        } catch FileSafetyError.notARegularFile(path: _, kind: let kind) {
            throw ParametersFileWriterError.destinationNotAFile(url.path, kind: kind.description)
        } catch FileSafetyError.fileChangedSinceWritten {
            throw ParametersFileWriterError.destinationChangedDuringWrite(url.path)
        }
    }

    /// The error for a markdown write that failed after the JSON file was
    /// written, after undoing the JSON write where that is possible.
    private static func markdownFailure(_ error: Error,
                                        markdownURL: URL,
                                        jsonURL: URL,
                                        writtenJSON: FileSafety.FileIdentity,
                                        jsonReplacedAFile: Bool) -> ParametersFileWriterError {
        let jsonOutcome: String
        if jsonReplacedAFile {
            jsonOutcome = "\(jsonURL.path) had already been replaced with the defaults and was left that way: "
                + "its earlier contents cannot be restored"
        } else {
            do {
                switch try FileSafety.removeOwnedItem(at: jsonURL, identity: writtenJSON) {
                case .removed:
                    jsonOutcome = "\(jsonURL.path), written moments before, was removed again, so nothing was written"
                case .alreadyGone:
                    jsonOutcome = "\(jsonURL.path), written moments before, was already gone when it was to be removed again"
                }
            } catch let removalError {
                jsonOutcome = "\(jsonURL.path), written moments before, could not be removed again "
                    + "(\(removalError.localizedDescription)) and is still there"
            }
        }
        return .markdownNotWritten(markdownURL.path, reason: error.localizedDescription, jsonOutcome: jsonOutcome)
    }
}

enum ParametersFileWriterError: LocalizedError, Equatable {
    case destinationExists(String)
    case destinationNotAFile(String, kind: String)
    case folderMissing(String)
    case jsonAndMarkdownSamePath(String)
    /// The file at the path was not the one checked a moment before; it was
    /// left alone.
    case destinationChangedDuringWrite(String)
    /// The markdown file failed after the JSON file was written; the second
    /// string says what became of the JSON file.
    case markdownNotWritten(String, reason: String, jsonOutcome: String)

    var errorDescription: String? {
        switch self {
        case .destinationExists(let path):
            return "\(path) already exists; pass --force to overwrite"
        case .destinationNotAFile(let path, let kind):
            return "\(path) exists and is not a regular file (\(kind)); refusing to replace it, even with --force"
        case .folderMissing(let path):
            return "\(path) names a folder that does not exist"
        case .jsonAndMarkdownSamePath(let path):
            return "\(path) would be both the JSON and the markdown file; name the .json file or a folder"
        case .destinationChangedDuringWrite(let path):
            return "\(path) changed while it was being written (another file took its place); it was left alone"
        case .markdownNotWritten(let path, let reason, let jsonOutcome):
            return "\(path) could not be written (\(reason)); \(jsonOutcome)"
        }
    }
}
