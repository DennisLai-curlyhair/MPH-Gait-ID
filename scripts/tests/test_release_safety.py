from __future__ import annotations

import importlib.util
import json
import subprocess
import unittest
from pathlib import Path, PureWindowsPath


ROOT = Path(__file__).resolve().parents[2]
IS_APP = (ROOT / "mph_gait_id").is_dir()


class ReleaseSafetyTest(unittest.TestCase):
    def ignored(self, paths):
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--stdin"],
            cwd=ROOT, input="\n".join(paths) + "\n",
            text=True, capture_output=True, check=False,
        )
        self.assertIn(result.returncode, (0, 1), result.stderr)
        return set(result.stdout.splitlines())

    def test_private_and_generated_paths_are_ignored(self):
        paths = [
            ".env", ".env.local", "nested/.env", "id_ed25519",
            "tls/private.pem", "gallery.sqlite3-wal", "gallery.sqlite3-shm",
            "backup.db-journal", "archive/checkpoints.tar.gz",
            "capture/raw_data.npy", "captures/person_rgb.png",
            "docs/RELEASE_GUIDE.md", "docs/RELEASE_GUIDE_zh.md",
        ]
        if IS_APP:
            paths.append("mph_gait_id/model_bundles/user_custom_123/bundle.yaml")
        else:
            paths.extend([
                "dataset/local/pointcloud/person_001/clear_data.npy",
                "dataset/local/projection/person_001/frame.png",
                "dataset_proj/person_001/frame.png",
                "dataset/PersonRecognitionWalking_29/person_001/frame.png",
            ])
        self.assertEqual(self.ignored(paths), set(paths))

    def test_public_files_and_approved_assets_remain_trackable(self):
        paths = ["README.md", "SECURITY.md", ".env.example", ".env.template"]
        if IS_APP:
            assets = subprocess.check_output(
                ["git", "ls-files", "*.pt"], cwd=ROOT, text=True
            ).splitlines()
            self.assertEqual(len(assets), 8)
            paths.extend(assets)
        else:
            paths.extend([
                "dataset/local/pointcloud/.gitkeep",
                "dataset/local/projection/.gitkeep",
                "mph-gait/checkpoints/checkpoint_manifest.csv",
                "docs/figures/performance.png",
                "dataset/metadata/fixed_special5_protocol.json",
            ])
        self.assertEqual(self.ignored(paths), set())

    def test_home_path_guard_is_portable_without_bare_marker_false_positive(self):
        spec = importlib.util.spec_from_file_location(
            "release_validator", ROOT / "scripts/validate_release.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        values = [
            "/".join(("", "home", "example", "capture.npy")),
            "/".join(("", "Users", "example", "capture.npy")),
            str(PureWindowsPath("C:/") / "Users" / "example" / "capture.npy"),
        ]
        for value in values:
            with self.subTest(path=value):
                self.assertIsNotNone(module.LOCAL_HOME_PATH.search(value))
                self.assertIsNotNone(module.LOCAL_HOME_PATH.search(json.dumps(value)))
        for value in ("/home/", "/Users/", "docs/UI_GUIDE.md", "data/gallery.sqlite3"):
            self.assertIsNone(module.LOCAL_HOME_PATH.search(value))


if __name__ == "__main__":
    unittest.main()
