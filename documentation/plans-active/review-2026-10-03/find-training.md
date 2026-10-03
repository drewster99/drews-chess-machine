find-training (partial, truncated in #5):
1 MED-HIGH: holdForThisRun (TrainingParameters:2262-2285) never released; GUI resume always holds random_seed_mode unseeded (saved: nil hard-coded, SessionController+Training:255-266) -> next fresh run in same launch uses held values (also pre-feature holds: illegal-mass weight 0, dropout 0, label smoothing 0, LR cycle off). [overlaps app#3]
2 MED: legal-mass grace spent during buffer refill after resume (LegalMassCollapseDetectorState:26-31; call SessionController+Training:2620-2624) because box seeded with steps=rs.trainingSteps (:75-78).
3 LOW-MED: entropy probe uses physical slot order (ReplayBufferAnalyzer:578,622,641) -> different positions after resume.
4 LOW-MED: GameSerialCounter:22 precondition(firstSerial>=0) reachable from lineage next_game_serial decode (LineageRecord:672 no range check).
5 LOW: inspectStored .int truncates n.intValue (7.9 -> 7) silently [truncated]
