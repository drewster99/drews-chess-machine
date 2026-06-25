import XCTest
@testable import DrewsChessMachine

/// `--replay-corpus` accepts either a filesystem path or a bare corpus ID
/// resolved under the `Corpora/` store.
final class CorpusArgResolutionTests: XCTestCase {

    private func makeCorpus(at dir: URL) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: dir.appendingPathComponent("corpus.json"))
    }

    func testResolvesAbsolutePath() throws {
        let base = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("carg-\(UUID().uuidString)", isDirectory: true)
        let corpus = base.appendingPathComponent("mycorpus", isDirectory: true)
        try makeCorpus(at: corpus)
        defer { do { try FileManager.default.removeItem(at: base) } catch { /* best-effort */ } }

        let resolved = DrewsChessMachineApp.resolveCorpusDirectory(corpus.path)
        XCTAssertEqual(resolved?.resolvingSymlinksInPath().path, corpus.resolvingSymlinksInPath().path)
    }

    func testResolvesBareIDUnderCorpora() throws {
        let corpora = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("corpora-\(UUID().uuidString)", isDirectory: true)
        let id = "20260101-000000-TESTID"
        let corpus = corpora.appendingPathComponent(id, isDirectory: true)
        try makeCorpus(at: corpus)
        defer { do { try FileManager.default.removeItem(at: corpora) } catch { /* best-effort */ } }

        // The bare ID resolves under the (injected) Corpora store.
        let resolved = DrewsChessMachineApp.resolveCorpusDirectory(id, corporaDir: corpora)
        XCTAssertEqual(resolved?.resolvingSymlinksInPath().path, corpus.resolvingSymlinksInPath().path)
    }

    func testUnknownArgResolvesToNil() {
        let emptyCorpora = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("corpora-\(UUID().uuidString)", isDirectory: true)
        XCTAssertNil(DrewsChessMachineApp.resolveCorpusDirectory("not-a-corpus-\(UUID().uuidString)",
                                                                 corporaDir: emptyCorpora))
    }
}
