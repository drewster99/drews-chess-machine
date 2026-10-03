find-ui:
1 HIGH: Build New Model crashes on negative Blocks(count): BlockGroupInitOptionsView -> groupHasSkipProjection(at:) -> NetworkArchitecture.blockRange/expandedBlocks traps (Range lower>upper, Array(repeating:count:<0)) before validate(); Neutral init button same path. Huge count allocates per render. From 67f5ef09.
2 MED: Invalid-settings sheet + auto-resume sheet both presented at launch (UpperContentView:1760/1856; sheets :1215/:1267); countdown (AutoResumeController:117-134) fires performResume after 30s even if its sheet isn't visible.
3 MED: resetInvalidStoredSetting (TrainingParameters:2173-2183) assigns live value via applyOne, overriding --parameters / resumed session values; sheet shown under --train too. Reset All stops at first failure.
4 LOW-MED: untouched out-of-range restored Int/Double fields (not in roundedTextFields) block popover Save (TrainingSettingsPopoverModel many lines; ArenaSettingsPopoverModel 189,199,348).
5 LOW: hardcoded Stepper ranges disagree with declarations: TrainingSettingsPopover:1206 batch 32...32768 (decl 65536); :1571 concurrency 1...256 (decl 8192); :2247 replayRatio 0.1...5 (decl 0.01...100); :2281 selfPlayDelay 0...3000 (decl 10000); live steppers may clamp & push.
6 LOW: Build sheet sets session.buildArchitecture/buildInitSeed and logs seed before refusal (UpperContentView:1238; SessionController:1073-1092).
7 minor: nonStandardInitLine @ViewBuilder helper func (ArchitectureDiagramView); O(G^2) architecture builds per render.
