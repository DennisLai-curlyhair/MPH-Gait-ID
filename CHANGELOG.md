# Changelog

## Unreleased

### Added
- Opt-in, pre-sampling enrollment foreground XYZ recordings with bounded background writing.
- Review-selected source commits, playback, storage limits and independent deletion.
- Bilingual source-library controls and operational documentation.

### Safety
- Atomic Gallery/source indexing, owner UID tracking, hashes and crash cleanup.
- Foreground recordings remain local; Gallery transfer archives still contain descriptors only.

## 0.2.0

### Added
- Portable Gallery export and import.
- Identity renaming and permanent fragment/person deletion.
- Explicit restoration of previously deleted Gallery records.
- Pre-operation backups and restoration history.

### Safety
- Restoring deleted identities requires unused target IDs.
- Model compatibility and duplicate checks remain enforced.
- Back up the Gallery database before upgrading.