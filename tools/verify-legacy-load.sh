#!/bin/sh
#
# verify-legacy-load.sh — on-demand validation that every real legacy .dcmmodel
# on disk resolves its archHash to a historical preset, rebuilds that
# architecture, loads its weights, and round-trips .dcmmodel -> safetensors ->
# reload bit-exact.
#
# This check is GPU-heavy and depends on real files, so it is gated OUT of the
# normal XCTest suite (LegacyDcmmodelLoadTests skips unless DCM_RUN_LEGACY_LOAD=1)
# to keep the suite under the build+run time cap. Run it explicitly here.
#
# If the env var doesn't propagate to the test runner in your toolchain, set
# DCM_RUN_LEGACY_LOAD=1 in the scheme's Test action → Environment Variables and
# run that test from Xcode instead.
#
set -eu
cd "$(dirname "$0")/.."
DCM_RUN_LEGACY_LOAD=1 xcodebuild test \
  -project DrewsChessMachine/DrewsChessMachine.xcodeproj \
  -scheme DrewsChessMachine \
  -only-testing:DrewsChessMachineTests/LegacyDcmmodelLoadTests \
  "$@"
