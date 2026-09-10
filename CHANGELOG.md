# Changelog

## Unreleased

### Added
- Multi-model source re-registration with per-target clip lengths, progress and cancellation.
- Transactional Gallery writes, pass-level deduplication and local job audit records.
- Synthetic three-model re-encoding/live-output and Gallery-transfer smoke test.
- Opt-in, pre-sampling enrollment foreground XYZ recordings with bounded background writing.
- Review-selected source commits, playback, storage limits and independent deletion.
- Bilingual source-library controls and operational documentation.

### Safety
- Source re-registration preserves owner UIDs, preprocessing profiles and model isolation.
- LidarGait++ transfer compatibility reads the flattened dimension from its nested part heads.
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