import XCTest
@testable import DrewsChessMachine

/// Decoding Lichess Bot API payloads (Lichess bot plan §4, E2, E17, E24,
/// E26, E28, E29).
///
/// Fixtures are the example payloads from the Lichess OpenAPI spec
/// (github.com/lichess-org/api, doc/specs/examples/), converted from the
/// spec's YAML flow style to strict JSON. Payloads captured from real games
/// are added alongside these in Phase 5.
final class LichessBotAPIModelsTests: XCTestCase {

    private func data(_ json: String) -> Data {
        Data(json.utf8)
    }

    // MARK: - Game stream

    /// doc/specs/examples/bot-streamBotGameState-gameFull.json.yaml
    private let gameFull = """
    {"id":"pG3WSP96","variant":{"key":"standard","name":"Standard","short":"Std"},"speed":"blitz","perf":{"name":"Blitz"},"rated":false,"createdAt":1789845924787,"white":{"id":"akeem","name":"Akeem","title":null,"rating":1523},"black":{"id":"bot0","name":"Bot0","title":"BOT","rating":1500,"provisional":true},"initialFen":"startpos","clock":{"initial":300000,"increment":0},"type":"gameFull","state":{"type":"gameState","moves":"","wtime":300000,"btime":300000,"winc":0,"binc":0,"status":"started"}}
    """

    func testDecodesGameFull() throws {
        guard case .gameFull(let full) = try LichessBotGameStreamLine.decode(data(gameFull)) else {
            return XCTFail("expected gameFull")
        }
        XCTAssertEqual(full.id, "pG3WSP96")
        XCTAssertEqual(full.variant.key.known, .standard)
        XCTAssertEqual(full.speed.known, .blitz)
        XCTAssertFalse(full.rated)
        XCTAssertEqual(full.clock?.initial, LichessBotMilliseconds(300_000))
        XCTAssertEqual(full.clock?.increment, LichessBotMilliseconds(0))
        XCTAssertEqual(full.white.id, "akeem")
        XCTAssertNil(full.white.title)
        XCTAssertEqual(full.black.title, "BOT")
        XCTAssertEqual(full.black.provisional, true)
        XCTAssertEqual(full.initialFen, "startpos")
        XCTAssertEqual(full.state.status.known, .started)
        XCTAssertEqual(full.state.status.isLive, true)
    }

    /// E2: `moves` is an empty string at the start — zero moves, not one
    /// empty token.
    func testEmptyMoveListIsZeroTokens() throws {
        guard case .gameFull(let full) = try LichessBotGameStreamLine.decode(data(gameFull)) else {
            return XCTFail("expected gameFull")
        }
        XCTAssertEqual(full.state.moveTokens, [])
    }

    func testInitialFenDefaultsToStartposWhenAbsent() throws {
        let withoutFen = gameFull.replacingOccurrences(of: "\"initialFen\":\"startpos\",", with: "")
        XCTAssertNotEqual(withoutFen, gameFull, "precondition: the field was removed")
        guard case .gameFull(let full) = try LichessBotGameStreamLine.decode(data(withoutFen)) else {
            return XCTFail("expected gameFull")
        }
        XCTAssertEqual(full.initialFen, "startpos")
    }

    /// doc/specs/examples/bot-streamBotGameState-gameState.json.yaml, plus
    /// a longer move list.
    func testDecodesGameStateAndSplitsMoves() throws {
        let json = #"{"type":"gameState","moves":"e2e4 e7e5  g1f3","wtime":299000,"btime":298500,"winc":2000,"binc":2000,"status":"started"}"#
        guard case .gameState(let state) = try LichessBotGameStreamLine.decode(data(json)) else {
            return XCTFail("expected gameState")
        }
        XCTAssertEqual(state.moveTokens, ["e2e4", "e7e5", "g1f3"])
        XCTAssertEqual(state.remaining(for: .white), LichessBotMilliseconds(299_000))
        XCTAssertEqual(state.remaining(for: .black), LichessBotMilliseconds(298_500))
    }

    /// E29: Lichess omits the draw/takeback flags when false.
    func testOmittedOfferFlagsMeanFalse() throws {
        let json = #"{"type":"gameState","moves":"e2e4","wtime":1,"btime":1,"winc":0,"binc":0,"status":"started","bdraw":true}"#
        guard case .gameState(let state) = try LichessBotGameStreamLine.decode(data(json)) else {
            return XCTFail("expected gameState")
        }
        XCTAssertFalse(state.isOfferingDraw(.white))
        XCTAssertTrue(state.isOfferingDraw(.black))
        XCTAssertFalse(state.isProposingTakeback(.white))
        XCTAssertFalse(state.isProposingTakeback(.black))
    }

    /// E28: an unknown status keeps its raw value and reports liveness as
    /// unknown rather than decoding as some known case.
    func testUnknownStatusIsPreservedNotFatal() throws {
        let json = #"{"type":"gameState","moves":"","wtime":1,"btime":1,"winc":0,"binc":0,"status":"someNewEnding"}"#
        guard case .gameState(let state) = try LichessBotGameStreamLine.decode(data(json)) else {
            return XCTFail("expected gameState")
        }
        XCTAssertNil(state.status.known)
        XCTAssertEqual(state.status.raw, "someNewEnding")
        XCTAssertNil(state.status.isLive)
    }

    func testEveryKnownStatusHasALiveness() {
        for status in LichessBotGameStatusName.allCases {
            let isLive = LichessBotOpenValue(status).isLive
            XCTAssertNotNil(isLive, "\(status)")
            XCTAssertEqual(isLive, status == .created || status == .started, "\(status)")
        }
    }

    func testDecodesChatLineAndOpponentGone() throws {
        guard case .chatLine(let chat) = try LichessBotGameStreamLine.decode(data(#"{"type":"chatLine","room":"player","username":"Bot0","text":"Good luck"}"#)) else {
            return XCTFail("expected chatLine")
        }
        XCTAssertEqual(chat.room.known, .player)
        XCTAssertEqual(chat.text, "Good luck")

        guard case .opponentGone(let gone) = try LichessBotGameStreamLine.decode(data(#"{"type":"opponentGone","gone":true,"claimWinInSeconds":50}"#)) else {
            return XCTFail("expected opponentGone")
        }
        XCTAssertTrue(gone.gone)
        XCTAssertEqual(gone.claimWinInSeconds, 50)
    }

    func testUnknownGameStreamLineTypeIsNotFatal() throws {
        guard case .unknown(let type) = try LichessBotGameStreamLine.decode(data(#"{"type":"somethingNew","x":1}"#)) else {
            return XCTFail("expected unknown")
        }
        XCTAssertEqual(type, "somethingNew")
    }

    // MARK: - Event stream

    /// doc/specs/examples/stream-gameStart.json.yaml
    func testDecodesGameStart() throws {
        let json = #"{"type":"gameStart","game":{"fullId":"wtRWmfWCSU1E","gameId":"wtRWmfWC","fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","color":"white","lastMove":"","source":"friend","status":{"id":20,"name":"started"},"variant":{"key":"standard","name":"Standard"},"speed":"blitz","perf":"blitz","rated":true,"hasMoved":false,"opponent":{"id":"aaron","username":"Aaron","rating":734},"isMyTurn":true,"secondsLeft":300,"rating":695,"compat":{"bot":false,"board":true},"id":"wtRWmfWC"}}"#
        guard case .gameStart(let game) = try LichessBotEvent.decode(data(json)) else {
            return XCTFail("expected gameStart")
        }
        XCTAssertEqual(game.gameId, "wtRWmfWC")
        XCTAssertEqual(game.color?.known, .white)
        XCTAssertEqual(game.status?.name.known, .started)
        XCTAssertEqual(game.opponent?.id, "aaron")
        XCTAssertEqual(game.isMyTurn, true)
    }

    /// doc/specs/examples/stream-gameFinish.json.yaml
    func testDecodesGameFinish() throws {
        let json = #"{"type":"gameFinish","game":{"fullId":"wtRWmfWCSU1E","gameId":"wtRWmfWC","fen":"rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2","color":"white","lastMove":"e7e5","source":"friend","status":{"id":31,"name":"resign"},"variant":{"key":"standard","name":"Standard"},"speed":"blitz","perf":"blitz","rated":true,"hasMoved":true,"opponent":{"id":"aaron","username":"Aaron","rating":734,"ratingDiff":32},"isMyTurn":false,"secondsLeft":300,"winner":"black","rating":695,"ratingDiff":-13,"compat":{"bot":false,"board":true},"id":"wtRWmfWC"}}"#
        guard case .gameFinish(let game) = try LichessBotEvent.decode(data(json)) else {
            return XCTFail("expected gameFinish")
        }
        XCTAssertEqual(game.status?.name.known, .resign)
        XCTAssertEqual(game.status?.name.isLive, false)
        XCTAssertEqual(game.winner?.known, .black)
        XCTAssertEqual(game.ratingDiff, -13)
    }

    /// doc/specs/examples/stream-gameStart-ai.json.yaml — E24: the opponent
    /// can be Lichess's AI, with a null id and no rating.
    func testDecodesAIOpponent() throws {
        let json = #"{"type":"gameStart","game":{"fullId":"YfjTIV43miXK","gameId":"YfjTIV43","fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","color":"white","lastMove":"","source":"ai","status":{"id":20,"name":"started"},"variant":{"key":"standard","name":"Standard"},"speed":"correspondence","perf":"correspondence","rated":false,"hasMoved":false,"opponent":{"id":null,"username":"Stockfish level 1","ai":1},"isMyTurn":true,"rating":2419,"compat":{"bot":false,"board":true},"id":"YfjTIV43"}}"#
        guard case .gameStart(let game) = try LichessBotEvent.decode(data(json)) else {
            return XCTFail("expected gameStart")
        }
        XCTAssertNil(game.opponent?.id)
        XCTAssertEqual(game.opponent?.ai, 1)
        XCTAssertNil(game.opponent?.rating)
    }

    /// doc/specs/examples/stream-challenge.json.yaml — E17: challenge time
    /// controls are in seconds.
    func testDecodesChallenge() throws {
        let json = #"{"type":"challenge","challenge":{"id":"iOskobMC","url":"https://lichess.org/iOskobMC","status":"created","challenger":{"name":"Adriana","patron":true,"patronColor":1,"id":"adriana","rating":548},"destUser":{"name":"Gabriela","flair":"objects.mobile-phone-with-arrow","id":"gabriela","rating":1459,"online":true},"variant":{"key":"standard","name":"Standard","short":"Std"},"rated":false,"speed":"correspondence","timeControl":{"type":"unlimited"},"color":"random","finalColor":"white","perf":{"icon":"","name":"Correspondence"}},"compat":{"bot":false,"board":true}}"#
        guard case .challenge(let challenge, let compat) = try LichessBotEvent.decode(data(json)) else {
            return XCTFail("expected challenge")
        }
        XCTAssertEqual(challenge.id, "iOskobMC")
        XCTAssertEqual(challenge.challenger.id, "adriana")
        XCTAssertEqual(challenge.timeControl.type.known, .unlimited)
        XCTAssertNil(challenge.timeControl.limit)
        XCTAssertEqual(challenge.color.known, .random)
        XCTAssertEqual(compat?.bot, false)
        XCTAssertNil(challenge.direction)

        let clock = #"{"type":"challenge","challenge":{"id":"c1","status":"created","challenger":{"id":"x","name":"X"},"destUser":null,"variant":{"key":"standard","name":"Standard"},"rated":true,"speed":"blitz","timeControl":{"type":"clock","limit":300,"increment":3,"show":"5+3"},"color":"white","direction":"in"}}"#
        guard case .challenge(let clocked, _) = try LichessBotEvent.decode(data(clock)) else {
            return XCTFail("expected challenge")
        }
        XCTAssertEqual(clocked.timeControl.limit, LichessBotSeconds(300))
        XCTAssertEqual(clocked.timeControl.increment, LichessBotSeconds(3))
        XCTAssertEqual(clocked.direction?.known, .incoming)
        XCTAssertNil(clocked.destUser)
    }

    /// doc/specs/examples/stream-challengeDeclined.json.yaml and
    /// stream-challengeCanceled.json.yaml
    func testDecodesChallengeCanceledAndDeclined() throws {
        let declined = #"{"type":"challengeDeclined","challenge":{"id":"iOskobMC","url":"https://lichess.org/iOskobMC","status":"declined","declineReason":"I'm not accepting challenges at the moment.","declineReasonKey":"generic"}}"#
        guard case .challengeDeclined(let reference) = try LichessBotEvent.decode(data(declined)) else {
            return XCTFail("expected challengeDeclined")
        }
        XCTAssertEqual(reference.id, "iOskobMC")

        let canceled = #"{"type":"challengeCanceled","challenge":{"id":"GCluxyas","status":"canceled"}}"#
        guard case .challengeCanceled(let canceledReference) = try LichessBotEvent.decode(data(canceled)) else {
            return XCTFail("expected challengeCanceled")
        }
        XCTAssertEqual(canceledReference.id, "GCluxyas")
    }

    func testUnknownEventTypeIsNotFatal() throws {
        guard case .unknown(let type) = try LichessBotEvent.decode(data(#"{"type":"brandNewEvent"}"#)) else {
            return XCTFail("expected unknown")
        }
        XCTAssertEqual(type, "brandNewEvent")
    }

    // MARK: - Account and token

    func testDecodesAccountAndBotTitle() throws {
        let json = #"{"id":"drewschessmachine","username":"DrewsChessMachine","title":"BOT","count":{"all":0,"rated":0,"draw":0,"loss":0,"win":0,"bookmark":0,"playing":0,"import":0,"me":0},"perfs":{"blitz":{"games":0,"rating":1500,"rd":500,"prog":0,"prov":true},"storm":{"runs":0,"score":0}},"createdAt":1790573862095}"#
        let account = try JSONDecoder().decode(LichessBotAccount.self, from: data(json))
        XCTAssertEqual(account.id, "drewschessmachine")
        XCTAssertTrue(account.isBot)
        XCTAssertEqual(account.count?.all, 0)
        XCTAssertEqual(account.perfs?["blitz"]?.rating, 1500)
        XCTAssertEqual(account.perfs?["blitz"]?.prov, true)
        XCTAssertNil(account.perfs?["storm"]?.rating, "puzzle modes decode with only shared fields")
    }

    func testTokenInfoScopes() throws {
        let json = #"{"userId":"drewschessmachine","scopes":"bot:play,challenge:read","expires":null}"#
        let info = try JSONDecoder().decode(LichessBotTokenInfo.self, from: data(json))
        XCTAssertEqual(info.scopeList, ["bot:play", "challenge:read"])
        XCTAssertNil(info.expires)
    }
}
