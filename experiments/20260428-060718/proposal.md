self_play_workers 48→40 (-17%) (streak=13). Fewer parallel workers — less GPU contention with checkpoint autosave I/O. Tests inverse direction of the 56-worker attempt that didnt help.
