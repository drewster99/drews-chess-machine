find-network:
1 MED: "same seed = same network on every machine" false for fresh nets: BN running stats calibrated by GPU forward (ChessMPSNetwork:130-137, 210-219); policy_pre_bn stats depend on --policy-tail-precision process flag; doc ChessMPSNetwork:22-26.
2 LOW-MED: MoveSampler still uses vForce/libm (vvexpf :148, powf :257, logf :288/:302, cosf :302) though DCMNormalMath:15-19 claims move sampling reproduces; stale TODO MoveSampler:51-54; sampleStandardNormal not switched to DCMRandom.nextStandardNormalPair.
3 LOW: awaitingWeightLoad doc (ChessNetwork:542-553) claims executionQueue-only; ChessTrainer reads on its own queue (:4482, :4628). internalLoadWeights (:1971-1980) clears flag without checking GPU command status.
4 LOW: test-only production code: InferenceNetworkFactory.build(arch:) (unlogged system seed), ChessNetwork.heInitData/glorotInitDataFCInOut/freshScaledNormals (SystemRandomNumberGenerator; only PolicyHeadCorrectnessTests), DCMRandom.jump() unused, DCMStream.corpusOrder reserved.
5 LOW: DCMRandom.swift:91-103 init(seed:) doc attached above seededFromSystem().
6 LOW: PolicyTailPrecision.processResolution uses CommandLine.arguments with preconditionFailure; launch check validates rawArgs ([] under XCTest).
