//
//  DCMRandomTests.swift
//  DrewsChessMachineTests
//
//  Golden-sequence tests for the seeded generator every reproducible random
//  draw will come from (determinism plan, Part A). The expected values were
//  computed once, outside this code: SplitMix64 and xoshiro256** outputs by
//  compiling the authors' reference C algorithms, and the bounded, unit,
//  shuffle and stream-derivation values by an independent Python
//  implementation whose raw xoshiro output matched the C reference. A failure
//  here means a seeded run would no longer reproduce, so these values must
//  never be edited to make a test pass — an intended algorithm change gets a
//  new derivation identifier instead.
//

import XCTest
@testable import DrewsChessMachine

final class DCMRandomTests: XCTestCase {

    // MARK: - SplitMix64

    func testSplitMix64FromSeedZeroMatchesTheReference() {
        var generator = DCMSplitMix64(state: 0)
        let outputs = (0..<8).map { _ in generator.next() }
        XCTAssertEqual(outputs, [
            0xe220a8397b1dcdaf, 0x6e789e6aa1b965f4, 0x06c45d188009454f, 0xf88bb8a8724c81ec,
            0x1b39896a51a8749b, 0x53cb9f0c747ea2ea, 0x2c829abe1f4532e1, 0xc584133ac916ab3c,
        ])
    }

    func testSplitMix64FromAFixedSeedMatchesTheReference() {
        var generator = DCMSplitMix64(state: 0x0123456789ABCDEF)
        let outputs = (0..<8).map { _ in generator.next() }
        XCTAssertEqual(outputs, [
            0x157a3807a48faa9d, 0xd573529b34a1d093, 0x2f90b72e996dccbe, 0xa2d419334c4667ec,
            0x01404ce914938008, 0x14bc574c2a2b4c72, 0xb8fc5b1060708c05, 0x8931545f4f9ea651,
        ])
    }

    func testSplitMix64MixIsTheFirstOutputFromThatState() {
        for state: UInt64 in [0, 1, 0x0123456789ABCDEF, .max] {
            var generator = DCMSplitMix64(state: state)
            XCTAssertEqual(DCMSplitMix64.mix(state), generator.next())
        }
    }

    // MARK: - xoshiro256**

    func testXoshiroFromAFixedStateMatchesTheReference() throws {
        var generator = try DCMRandom(s0: 1, s1: 2, s2: 3, s3: 4)
        let outputs = (0..<16).map { _ in generator.next() }
        XCTAssertEqual(outputs, [
            0x0000000000002d00, 0x0000000000000000, 0x000000005a007080, 0x10e0000000009d80,
            0x10e0b61ce1009d80, 0x0870021ce143ad00, 0xe071c3c2e143f089, 0x75a1690ef7a20380,
            0x9309685b465c23f9, 0x284f3cc2e13e3c88, 0xc8d749005a413820, 0x1194b410fef20904,
            0xb54a54470263b28c, 0x959e65495daf641c, 0xe561ccecea17f527, 0xd7713c78965a463c,
        ])
        generator.jump()
        let afterJump = (0..<4).map { _ in generator.next() }
        XCTAssertEqual(afterJump, [
            0x5c874fec44783b77, 0x17bcd9b08580dd16, 0x9ca7f9375f7dbeb2, 0x24caff1483ddd1fa,
        ])
    }

    func testSeedingExpandsOneWordThroughSplitMix64() {
        var generator = DCMRandom(seed: 42)
        let outputs = (0..<8).map { _ in generator.next() }
        XCTAssertEqual(outputs, [
            0x15780b2e0c2ec716, 0x6104d9866d113a7e, 0xae17533239e499a1, 0xecb8ad4703b360a1,
            0xfde6dc7fe2ec5e64, 0xc50da53101795238, 0xb82154855a65ddb2, 0xd99a2743ebe60087,
        ])
    }

    func testAnAllZeroStateIsRefused() {
        XCTAssertThrowsError(try DCMRandom(s0: 0, s1: 0, s2: 0, s3: 0)) { error in
            XCTAssertEqual(error as? DCMRandomError, .allZeroState)
        }
    }

    // MARK: - Bounded draws

    func testBoundedDrawsMatchTheReference() {
        let expected: [(UInt64, [UInt64])] = [
            (1, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            (2, [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1]),
            (3, [0, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 0, 2, 0, 2, 2]),
            (7, [0, 2, 4, 6, 6, 5, 5, 5, 5, 4, 4, 2, 5, 2, 4, 6]),
            (4864, [407, 1843, 3307, 4497, 4824, 3744, 3498, 4134, 3703, 2837, 3319, 1413, 3896, 1563, 3459, 4269]),
            ((1 << 32) + 1, [
                360188718, 1627707782, 2920764210, 3971525959, 4259765376, 3306005809, 3089192070, 3650758468,
                3270078067, 2505466208, 2931112756, 1248451480, 3440373161, 1380452453, 3054365756, 3769981545,
            ]),
            (.max, [
                1546998764402558741, 6990951692964543101, 12544586762248559008, 17057574109182124192,
                18295552978065317475, 14199186830065750583, 13267978908934200753, 15679888225317814406,
                14044878350692344957, 10760895422300929084, 12589033428110817648, 5362058279183681892,
                14776290213336893109, 5928998142081247041, 13118401031821625292, 16191947441114085369,
            ]),
        ]
        for (bound, values) in expected {
            var generator = DCMRandom(seed: 42)
            let drawn = (0..<16).map { _ in generator.nextBounded(bound) }
            XCTAssertEqual(drawn, values, "bound \(bound)")
        }
    }

    func testIntBoundedDrawsAgreeWithTheUInt64Form() {
        var asUnsigned = DCMRandom(seed: 42)
        var asInt = DCMRandom(seed: 42)
        for _ in 0..<64 {
            XCTAssertEqual(UInt64(asInt.nextBounded(4864)), asUnsigned.nextBounded(4864))
        }
    }

    func testBoundedDrawsOverThreeValuesAreUniform() {
        var generator = DCMRandom(seed: 0x0123456789ABCDEF)
        let drawCount = 3_000_000
        var counts = [0, 0, 0]
        for _ in 0..<drawCount {
            counts[Int(generator.nextBounded(UInt64(3)))] += 1
        }
        let expected = Double(drawCount) / 3
        let chiSquare = counts.reduce(0.0) { sum, count in
            let difference = Double(count) - expected
            return sum + difference * difference / expected
        }
        // Two degrees of freedom: anything below this bound is unremarkable
        // for a uniform source; a biased reduction lands far above it.
        XCTAssertLessThan(chiSquare, 30, "counts \(counts)")
    }

    // MARK: - Unit draws

    func testUnitDoubleDrawsMatchTheReference() {
        var generator = DCMRandom(seed: 42)
        let bits = (0..<8).map { _ in generator.nextUnitDouble().bitPattern }
        XCTAssertEqual(bits, [
            0x3fb5780b2e0c2ec0, 0x3fd84136619b444e, 0x3fe5c2ea66473c93, 0x3fed9715a8e0766c,
            0x3fefbcdb8ffc5d8b, 0x3fe8a1b4a6202f2a, 0x3fe7042a90ab4cbb, 0x3feb3344e87d7cc0,
        ])
    }

    func testUnitFloatDrawsMatchTheReference() {
        var generator = DCMRandom(seed: 42)
        let bits = (0..<8).map { _ in generator.nextUnitFloat().bitPattern }
        XCTAssertEqual(bits, [
            0x3dabc058, 0x3ec209b2, 0x3f2e1753, 0x3f6cb8ad, 0x3f7de6dc, 0x3f450da5, 0x3f382154, 0x3f599a27,
        ])
    }

    func testUnitDrawsStayInTheHalfOpenUnitInterval() {
        var generator = DCMRandom(seed: 7)
        for _ in 0..<1_000_000 {
            let double = generator.nextUnitDouble()
            XCTAssertTrue(double >= 0 && double < 1)
            let float = generator.nextUnitFloat()
            XCTAssertTrue(float >= 0 && float < 1)
        }
    }

    // MARK: - Shuffle

    func testStableShuffleMatchesTheReference() {
        var generator = DCMRandom(seed: 42)
        var values = Array(0..<10)
        values.stableShuffle(using: &generator)
        XCTAssertEqual(values, [9, 1, 4, 2, 8, 7, 6, 5, 3, 0])
    }

    // MARK: - Codable state

    func testStateRoundTripsThroughJSONWithDecimalStringWords() throws {
        var generator = DCMRandom(seed: 0x0123456789ABCDEF)
        for _ in 0..<5 { _ = generator.next() }
        let data = try JSONEncoder().encode(generator)
        let object = try JSONSerialization.jsonObject(with: data)
        guard let words = object as? [String: Any] else {
            XCTFail("state did not encode as a JSON object: \(String(decoding: data, as: UTF8.self))")
            return
        }
        XCTAssertEqual(Set(words.keys), ["s0", "s1", "s2", "s3"])
        for (key, value) in words {
            XCTAssertTrue(value is String, "\(key) is not a decimal string")
        }
        let restored = try JSONDecoder().decode(DCMRandom.self, from: data)
        XCTAssertEqual(restored, generator)
        var continued = generator
        var fromRestored = restored
        XCTAssertEqual((0..<8).map { _ in fromRestored.next() }, (0..<8).map { _ in continued.next() })
    }

    func testAWordAboveTwoToTheFiftyThirdSurvivesTheRoundTrip() throws {
        let large: UInt64 = (1 << 53) + 1
        let generator = try DCMRandom(s0: large, s1: .max, s2: 3, s3: 4)
        let restored = try JSONDecoder().decode(DCMRandom.self, from: JSONEncoder().encode(generator))
        XCTAssertEqual(restored, generator)
    }

    func testDecodingRefusesAnAllZeroState() {
        let json = #"{"s0":"0","s1":"0","s2":"0","s3":"0"}"#
        XCTAssertThrowsError(try JSONDecoder().decode(DCMRandom.self, from: Data(json.utf8)))
    }

    func testDecodingRefusesANumericWord() {
        let json = #"{"s0":1,"s1":"2","s2":"3","s3":"4"}"#
        XCTAssertThrowsError(try JSONDecoder().decode(DCMRandom.self, from: Data(json.utf8)))
    }

    func testDecodingRefusesAWordThatIsNotAUInt64() {
        for word in ["-1", "18446744073709551616", "0x10", "one", ""] {
            let json = #"{"s0":"\#(word)","s1":"2","s2":"3","s3":"4"}"#
            XCTAssertThrowsError(try JSONDecoder().decode(DCMRandom.self, from: Data(json.utf8)), "word \(word)")
        }
    }

    // MARK: - Stream derivation

    func testStableHashIsTheFirstEightBytesOfSHA256BigEndian() {
        XCTAssertEqual(DCMRandomStreams.stableHash64("sampler"), 0xb7a56c61657d4e5c)
        XCTAssertEqual(DCMRandomStreams.stableHash64("init"), 0xbb54068aea85faa7)
        XCTAssertEqual(DCMRandomStreams.stableHash64("init/block3_conv1_weights"), 0x5574bd09905b77b4)
        XCTAssertEqual(DCMRandomStreams.stableHash64("selfplay.game.17"), 0x26ee28a666a02a35)
    }

    func testChildSeedsArePinned() {
        let parent: UInt64 = 0x0123456789ABCDEF
        XCTAssertEqual(DCMRandomStreams.childSeed(parent: parent, name: "sampler"), 0x4caf3a147c394e4b)
        XCTAssertEqual(DCMRandomStreams.childSeed(parent: parent, name: "init"), 0xa0beee1afc11ca41)
        XCTAssertEqual(DCMRandomStreams.childSeed(parent: parent, name: "init/block3_conv1_weights"), 0x4fa0f80b22c803eb)
        XCTAssertEqual(DCMRandomStreams.childSeed(parent: parent, name: "selfplay.game.17"), 0xf5eea257a466669a)
    }

    func testAddingAStreamDoesNotChangeAnExistingChildSeed() {
        let parent: UInt64 = 0x0123456789ABCDEF
        let before = DCMRandomStreams.childSeed(parent: parent, name: "sampler")
        for name in ["arena.3.game.9", "a-stream-added-later", "probe.entropy_by_bucket.1000"] {
            _ = DCMRandomStreams.childSeed(parent: parent, name: name)
        }
        XCTAssertEqual(DCMRandomStreams.childSeed(parent: parent, name: "sampler"), before)
        XCTAssertEqual(before, 0x4caf3a147c394e4b)
    }

    func testStreamNamesAreTheCatalogSpellings() {
        XCTAssertEqual(DCMStream.sampler.name, "sampler")
        XCTAssertEqual(DCMStream.dropout.name, "dropout")
        XCTAssertEqual(DCMStream.selfPlayGame(serial: 17).name, "selfplay.game.17")
        XCTAssertEqual(DCMStream.arenaGame(arenaIndex: 3, gameIndex: 9).name, "arena.3.game.9")
        XCTAssertEqual(DCMStream.trainVsUciGame(serial: 5).name, "vsuci.game.5")
        XCTAssertEqual(DCMStream.probe(name: "entropy_by_bucket", trainerStep: 1000).name, "probe.entropy_by_bucket.1000")
        XCTAssertEqual(DCMStream.corpusOrder(epoch: 2).name, "corpus.order.2")
    }

    func testAStreamsGeneratorIsSeededByItsChildSeed() {
        let streams = DCMRandomStreams(masterSeed: 0x0123456789ABCDEF)
        var fromStreams = streams.generator(.selfPlayGame(serial: 17))
        var direct = DCMRandom(seed: 0xf5eea257a466669a)
        XCTAssertEqual((0..<8).map { _ in fromStreams.next() }, (0..<8).map { _ in direct.next() })
    }

    func testStandardLibraryUsingAPIsAcceptTheGenerator() {
        var generator = DCMRandom(seed: 1)
        let value = Int.random(in: 0..<10, using: &generator)
        XCTAssertTrue((0..<10).contains(value))
    }
}
