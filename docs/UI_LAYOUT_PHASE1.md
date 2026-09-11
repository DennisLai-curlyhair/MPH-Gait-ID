# UI Layout and State Preservation

## Scope

This Conference application update keeps the existing Tkinter/ttk interface.
It does not change checkpoint files, model inputs, matching rules, inference
contracts, storage schemas, dependencies, or the release version. No data migration
or re-enrollment is required. The journal worktree is separate.

## Changes

- Navigation tabs and static controls reserve space for both English and
  Traditional Chinese. Tables retain enough column width for their headings.
- Locale changes update presentation in place. Person IDs, names, paths, selected
  models, thresholds, recordings, and active operation objects remain unchanged.
- Revisiting Realtime or Performance refreshes Gallery counts without resetting
  settings for the same bundle. Changing the bundle, checkpoint hash, or bundle
  model/data configuration still reapplies that bundle's defaults.
- Realtime has a draggable sidebar boundary. Offline preview/results and
  Performance settings/telemetry/results have useful initial proportions and
  minimum visible areas. Long results and paths wrap within their allocated space.
- Settings and preview tools support scrolling when the window is small. Benchmark
  start/stop/export controls remain outside the scrolling settings. Result tables
  provide horizontal as well as vertical scrolling.
- Gallery refresh preserves the selected person and the same logical pass, even
  when row ordering changes. Changing person/model/T does not select a different
  person's similarly numbered pass.
- Source refresh preserves surviving selections, the current pass, frame, zoom,
  and list position. Leaving the Sources tab pauses playback but keeps its position.
  A deleted source is cleared; returning does not restart playback automatically.

View state is retained within the current application session. Persistent layout
preferences and broader workflow redesign are not part of this update. Dynamic
status messages can still change height when their contents need wrapping.

## Local Acceptance Checks

1. Try 1380x860 and 980x640 windows, then Windows 125% display scaling. Drag the
   splitters; confirm the preview and result areas remain accessible by resizing
   and scrolling. Confirm the source-transfer controls remain reachable.
2. Select a model and change T, threshold, detector weights, and a person's name.
   Switch English/Traditional Chinese repeatedly and revisit every tab. None of
   those values should reset. Explicitly choosing a different bundle should still
   apply its defaults.
3. In Gallery, select a person and pass and scroll the list. Refresh and switch
   languages. Verify that the same pass remains selected; repeat after renaming
   the person or removing another pass.
4. In Sources, select several recordings, choose a pass, seek and zoom. Refresh,
   switch tabs, and return. The position should remain and playback should be
   paused. Delete a selected source and verify no stale preview remains.
5. With an Azure Kinect connected, check locale switching during live recognition,
   enrollment capture/review, and benchmarking. Confirm acquisition continues and
   result reporting remains responsive. Hardware checks must be performed locally.
6. Recheck Gallery and source archive export/import with temporary test identities.
   Confirm their conflict preview and confirmation steps have not changed.

Run regression checks from the repository root in the application environment:

```bash
python -m unittest discover -s mph_gait_id/tests -q
python -m unittest discover -s scripts/tests -q
python scripts/validate_release.py
```

GUI tests need a graphical display and otherwise report skips. Automated GUI tests
use temporary databases and do not open the camera or modify registered identities.
