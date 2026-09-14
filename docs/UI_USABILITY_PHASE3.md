# UI Phase Three: Layout Preferences and Browsing

## Scope

This Conference application update improves the existing Tkinter/ttk UI.
It does not change models, weights, inference settings, recognition thresholds,
Gallery schemas, compatibility keys, or dependencies. Existing galleries and
foreground source recordings require no migration or re-enrollment.

## Layout Preferences

On normal application exit, the UI saves window dimensions and position,
supported pane proportions, principal table column widths, and expanded/collapsed
sections. Language already persists through the existing language setting.
The circular-arrow button next to the language selector resets the layout after
confirmation. It does not reset models, thresholds, identities or running jobs.

Layout preferences are stored in `settings_layout.json` next to the existing
`settings.json`, outside the repository by default:

| Platform | Default directory |
|---|---|
| Windows | `%APPDATA%/mph_gait_id/` |
| Linux | `$XDG_CONFIG_HOME/mph_gait_id/`, or `~/.config/mph_gait_id/` |
| macOS | `~/Library/Application Support/mph_gait_id/` |

`MPH_GAIT_ID_SETTINGS` overrides the language settings path. The layout file uses
the same directory and adds `_layout` to that file's stem. For example,
`test-settings.json` uses `test-settings_layout.json`. Separate paths can isolate
test installations. A corrupt, oversized or unsupported layout file falls back
to defaults. Writes use atomic replacement; a write failure is reported without
altering the Gallery.

Only named UI properties are saved. Search terms, selected people, paths to
recordings/checkpoints, camera state, recognition parameters and enrollment
sessions are not included. Layout restoration never starts a device or resumes
registration. Window geometry is clamped to the primary screen after a monitor
change; Windows maximization is restored where supported. On a smaller display,
minimum readable panes and table headings take precedence over exact saved sizes.
Unvisited panes retain saved proportions until first shown. Abrupt termination
may lose layout changes since the last normal exit, not Gallery data.

## Basic and Advanced Controls

Real-time source selection, person-detector type, model bundle, frame length,
stride and compute device remain visible. Detector weight paths and less common
matching/replay settings have collapsible sections. Offline unknown-rejection
settings are also collapsible. These sections only control visibility.

The real-time readiness summary retains detector/settings status and provides
`Show settings` buttons that expand and scroll to the corresponding controls.
Runtime configuration locks remain enforced while capture or review is pending.
Expanding a section does not unlock its settings.

## Gallery and Source Browsing

- Search the current Gallery scope or source library by Person ID or name,
  using case-insensitive substring matching.
- Click a column heading to sort; click again to reverse the order. Counts and
  source sizes sort numerically; IDs use natural ordering (`P2` before `P10`).
- The status line displays visible/total rows, selected rows and current sort.
- Clear the query using `Clear` or Escape while the search field has focus.
- Search and sorting survive refresh and language changes within the session,
  but are not restored after restarting the application.

Rows retain database/source identifiers. Sorting does not change selection or
the target of an operation. Filtering a selected row out clears its selection;
hidden selections are not exported, registered or deleted. Clearing the search
does not silently reselect hidden rows. Existing model/T/processing-version
scope and delete/import confirmations remain unchanged. Candidate rank tables
are not user-sortable and retain the recognizer's ordering.

## Preview Controls

Offline, source-library and real-time point-cloud views use shared symbol buttons
and bilingual hover/keyboard-focus tooltips for rotation, mirroring, pan and reset.
Source playback and offline playback use consistent previous/next/play controls.
Frame counters remain visible. Real-time orientation controls are collapsible;
zoom remains available in the preview header. Source playback retains its zoom
slider. Reset affects the preview only.

These operations do not alter stored coordinates, sampling, model input or
Gallery descriptors. Source/realtime views retain their fixed physical span;
offline views retain their existing fitted display. Compact viewports fit images
without cropping them merely to satisfy a minimum image size. Destructive Gallery
and source-delete commands are visually differentiated; all existing safeguards
remain required. Installed Traditional Chinese UI fonts are preferred where
available; table row height accounts for the active default font. Live/source
point counts stay outside the image instead of covering the preview.

## Local Acceptance

Back up the complete `data/` directory and use test identities.

1. Resize the window, drag panes and table columns, and expand advanced controls.
   Close normally and reopen. Check saved layout and language, with no capture
   started and no identity/session restored automatically.
2. Reset the layout after changing T and thresholds. Only the layout should reset.
   Test at 980x640, maximized, and Windows 125%/150% scaling.
3. Search by ID/name, reverse numeric and ID sorting, refresh and switch language.
   Check selected identities and source UUIDs remain correct. Filter out a selected
   source and verify it is no longer selected for export or model registration.
4. Enter invalid settings or detector paths, collapse the section and use the
   readiness links to locate it. Expanding during capture must not unlock inputs.
5. Play saved foreground recordings. Rotate, mirror, pan, zoom and reset; compare
   the same frame after reset. Switch language and inspect tooltips. Check both
   RGB and point-cloud previews remain framed in a small real-time viewport.
6. Repeat guided enrollment/review/save, recognition, multi-model registration,
   Gallery/source transfer and deletion using test data. These workflows should
   behave as before.

```bash
python -m unittest discover -s mph_gait_id/tests -q
python -m unittest discover -s scripts/tests -q
python scripts/validate_release.py
```

GUI tests require a graphical display. Server screenshots and regression tests
use temporary databases and synthetic point clouds, not a connected Azure Kinect.
Native Windows scaling and device behavior require Local verification.
