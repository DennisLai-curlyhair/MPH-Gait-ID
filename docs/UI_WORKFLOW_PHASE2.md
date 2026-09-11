# UI Workflow and Feedback

## Scope

This update improves the Conference application's operating feedback. Model
weights, embedding compatibility keys, Gallery schema, recognition thresholds,
sampling rules, and installation dependencies are unchanged. Existing Gallery
embeddings and foreground recordings do not require migration or re-enrollment.

## Before Starting

The Real-time sidebar includes a Readiness section covering the source,
model/checkpoint, detector, compatible Gallery, enrollment identity, and settings.
Missing local paths, incomplete identity fields, conflicting names, and invalid
numeric settings disable Start. Correcting the input updates the checklist.

The checklist is not a successful inference or camera-access test. Azure Kinect
is opened and checked by the capture worker on every run. The optional device
check also runs in the background. Model loading and detector verification can
still fail after a local checkpoint has been found. SAM path checks do not hash
large files on the UI thread; full hash verification remains mandatory in the
background detector before loading.

An empty compatible Gallery is a warning in Real-time mode: sensor preview and
foreground extraction can run, but identification is unavailable. Enrollment does
not require an existing Gallery. Offline identification checks Gallery availability
before encoding. Compatibility continues to depend on the selected model,
checkpoint, frame length, and existing preprocessing contract.

## Live Feedback

- Source opening and model loading are distinguished from frame collection.
- Waiting for a person, strict-mode multiple-person rejection, insufficient
  foreground points, pending stability, unknown identity, and empty Gallery have
  separate messages. Sensor problems take precedence over an earlier candidate.
- The progress bar represents the current frame buffer, not confidence, task
  completion, or identity accuracy. The existing similarity, margin and stability
  rules are not recomputed by the interface.
- Short windows keep RGB and point-cloud previews side by side; a stacked view
  is used only when sufficient vertical space is available.
- Capture settings remain locked while the worker starts, runs, stops, or awaits
  enrollment review. Stopping runs in the background; controls are not released
  until the capture worker has actually exited.
- Language changes update presentation without changing names or operational
  state. Raw diagnostics and third-party error messages may retain their original
  language.

## Guided Enrollment

1. Enter the identity and start the guided session. Start pass remains disabled
   until the detector and gait model have loaded and the source produces frames.
2. Start pass requests capture. Warm-up requires a valid person; requesting a pass
   is not the same as having recorded one. Repeated clicks cannot queue duplicates.
3. End or discard a pass waits for acknowledgement from the capture worker before
   re-enabling Start pass. Finish & review ends capture and opens review.
4. Review lists the selected passes and candidate embedding count. With no
   selection, Commit is disabled. Closing and reopening review preserves selection
   during this application session.
5. Committing locks review controls and shows ongoing activity. Gallery is updated
   only when the existing commit operation succeeds. A failed commit retains the
   review for correction or retry. Continuing capture preserves existing passes.

The four labels (Prepare, Capture, Review, Save) indicate the current phase.
Stop cancels capture; it is not a substitute for Finish & review or Commit.
Pending review must be committed, continued, or abandoned before a new session.

## Batch Operations

Multi-model source enrollment separates Model results from the Activity log.
The status line displays current work; completion lists each model's outcome and
embedding additions. Cancelled/failed jobs are not labeled successful; the existing
all-or-nothing registration transaction is retained.

Gallery and source transfer distinguish package verification, export and import.
Existing conflict previews, confirmation dialogs, backups and cancellation
behavior are retained. Source progress displays the reported frame count; stages
without a reliable work total remain indeterminate.

## Local Acceptance

Use a separate test clone and back up local data before testing.

- Check both languages at 980x640, maximized, and Windows display scaling.
- Leave the name empty; select a missing replay/detector path; restore valid input.
  Confirm that Readiness identifies the cause and Start recovers.
- Start without a Gallery. Confirm preview works but no identity is accepted.
  Enroll or import compatible embeddings and check recognition again.
- Disconnect Kinect; start and check the device. The UI should stay responsive
  and report the device failure. Reconnect and retry.
- Exercise waiting person, warm-up, start/end/discard, review, continued capture,
  commit, and abandon. Switch languages during capture and stopping.
- Unselect all review passes; close/reopen review; confirm no selection is silently
  restored. Test a normal successful commit and verify Gallery counts.
- Run multi-model enrollment and source/Gallery export/import. Verify summaries,
  cancellation, duplicates and conflict handling using test identities.
- Check that a running device probe or stopping worker blocks conflicting pages.

Automated checks (GUI tests require a display):

```bash
python -m unittest discover -s mph_gait_id/tests -q
python -m unittest discover -s scripts/tests -q
python scripts/validate_release.py
```

Server tests use temporary databases and simulated capture workers. They do not
replace Windows/Azure Kinect validation. Runtime shutdown still depends on the
SDK/backend returning; the UI must not claim the device is free while it is busy.
