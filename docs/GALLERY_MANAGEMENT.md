# Gallery Management

[Documentation index](README.md) | [Traditional Chinese](GALLERY_MANAGEMENT_zh.md)

Open **Gallery Manager** after stopping recognition, enrollment and performance
measurement. Commit or abandon any pending enrollment review before editing or
transferring Gallery. Model weights and encoding settings are not changed by
these operations.

## Operations and Scope

| Operation | Scope | Retained data |
|---|---|---|
| Rename person (all models) | Display name for the selected Person ID, across all models | ID, embeddings, active/inactive states and history |
| Deactivate / Reactivate selected pass | Selected person, model key, session and pass | All stored vectors |
| Permanently delete selected fragment | Selected person, model key, source, session, pass and direction; includes inactive vectors | Person record, other fragments and other models |
| Delete person and ALL model embeddings | Selected person and all their active/inactive vectors, regardless of model filters | Other people, model records and weights |
| Export all Gallery / Import Gallery | See [Gallery transfer](GALLERY_TRANSFER.md) | Existing compatible local records are merged, not replaced |

The Offline page's **Deactivate this identity's embeddings for the current
checkpoint** is a soft operation. It does not permanently delete vectors or
affect other model bundles. Use Gallery Manager for permanent deletion.

**All people (all models, including empty entries)** expands only the left
identity table. The top summary and right fragment table still follow the model,
frame-length and point-cloud-processing filters. Horizontal scrollbars expose
source IDs and other columns in narrower windows. Imported source IDs are
identifiers, not playable media paths.

## Rename or Delete

1. Select the identity. Enable **All people** if it has no remaining embeddings
   or belongs to a different model.
2. For a rename, enter the new display name. Person ID is unchanged. Subsequent
   enrollment must use that name; update any already-filled enrollment fields.
3. For fragment deletion, select one row on the right and inspect its source,
   session, pass, direction and active/inactive counts before confirming.
4. Whole-person deletion requires typing the exact, case-sensitive Person ID.
   This operation is not restricted to the currently selected model.
5. Check the updated counts and the pre-edit backup path in the success dialog.

Legacy sources without session/pass IDs can be permanently deleted by source
row. Pass activation still requires both IDs. Deleting the last fragment retains
the person and enables **All people** so the empty identity stays manageable.
Whole-person deletion frees the visible ID; using a new ID for a different person
is still preferable because historical reports retain the original identity.

## Backups and Recovery

Every rename or permanent delete creates a verified SQLite snapshot before
changing data. Backups are stored beside the configured database:

```text
data/backups/gallery_before_edit_<timestamp>_<unique-id>.sqlite3
```

SQLite's backup API includes committed WAL data. The write transaction checks
that the selected records have not changed since confirmation. Backup failure
cancels the edit; a database failure rolls back the entire edit, including
transfer identity bookkeeping. Backup and deletion run synchronously, so large
databases may briefly pause the UI. Do not force-close the application.

Permanent deletion removes rows from the active database, not from original
point clouds, sessions, reports, archives or existing backups. It is **not secure
erasure**. Backups contain names and biometric vectors, are not encrypted, and
are not automatically pruned. They are excluded from Git; protect storage access
and define a retention policy.

An exported `.mphgallery` can selectively restore archived records through
explicit import confirmation (below). It does not restore source media or undo
all later changes. For whole-database recovery, close all database connections,
preserve the current database and its sidecars, then restore the pre-edit SQLite
backup. A full snapshot restore also removes registrations made after that snapshot.

## Restore Deleted Records

1. Open **Import Gallery** and select an archive made before deletion.
2. Review the People and Models tabs. Deleted records are skipped by default.
3. Enable **Restore previously deleted records**.
4. A deleted person's original ID is selected if unused. If occupied, use
   **New ID** to choose an unused ID. Deleted identities cannot be merged into an
   existing person, even when the name matches. Other unresolved conflicts remain skipped.
5. Confirm the selected people and compatible feature counts, then confirm import.
   The importer creates a backup before writing.
6. Check the restored counts and verify recognition with the matching model.

If the person still exists and only fragments were deleted, restoration adds those
archived features to the linked person without replacing the local name or status.
Restored rows use the archive's active/inactive state; already present inactive
rows stay inactive. Duplicate records are not added again.

Only selected records present in the archive and compatible with installed models
are restored. Missing models leave their feature deletion markers intact; install
the matching bundle and import again with restoration enabled. An empty person
can be restored without features. Restoring a deleted person may therefore restore
the person record even when no compatible embeddings are currently available.

The local `gallery_transfer_restorations` table records each restored UID, target
ID, archive SHA256, deletion time and restoration time. Deletion markers are removed
only for successfully restored UIDs in the same transaction. Errors roll back
records, markers and audit entries together. This history is not tamper-proof.

## Interaction With Transfer

- Renaming preserves the portable identity. Importing an older archive keeps
  the current local name and does not reactivate inactive vectors.
- A permanent fragment delete records its known portable identities. Reimport
  skips those vectors unless explicit restoration is enabled.
- A deleted person defaults to **Skip**. Explicit restoration requires an unused
  target ID; it never silently merges into a reused ID. Restored UIDs keep their
  portable identity, so later imports are deduplicated.
- A newly registered person using the same visible ID receives a different
  portable identity when exported. Another PC that still has the old person
  receives an ID conflict, not an automatic identity match.
- Deletion history is local and is not exported as a synchronization command.
  Existing archives/backups and other PCs are not revoked or erased. A clean PC
  with no deletion history can still import an old archive.
- Direct SQLite edits and third-party archives with newly assigned identities
  are not a supported way to bypass these protections. Use the management UI
  and preserve its provenance tables.

The archive format remains version 1. Existing embeddings, model keys and
checkpoint compatibility do not change. Additional local bookkeeping tables
are initialized when management/transfer is first used; existing Gallery rows
are not deleted or renamed automatically.

## Verification

Run from the repository root:

```bash
python -m unittest discover -s mph_gait_id/tests -v
python -m unittest discover -s scripts/tests -v
```

Tests use temporary databases and synthetic vectors. Graphical tests skip when
Tk cannot open a display. Validate both UI languages and actual device workflows
on a graphical computer using a separate test database before deployment.
