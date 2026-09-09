from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch
import yaml

from mph_gait_id.model_adapter import ModelAdapter
from mph_gait_id.model_store import ModelStore, sha256_file


PACKAGE = Path(__file__).resolve().parents[1]
METHODS = ("pointnet_tmax", "mph_gait", "lidargaitpp")


class Final24BundlesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.store = ModelStore()
        cls.assets = yaml.safe_load((PACKAGE / "assets/manifest.yaml").read_text())

    def test_assets_and_training_metadata(self) -> None:
        for method in METHODS:
            with self.subTest(method=method):
                bundle = self.store.get(f"{method}_final24_len15_seed2")
                asset_id = f"{method}_final24_checkpoint"
                for profile in ("offline-demo", "realtime-yolo", "final24", "all"):
                    self.assertIn(asset_id, self.assets["profiles"][profile])
                asset = self.assets["assets"][asset_id]
                self.assertEqual(bundle.checkpoint, (PACKAGE / asset["target"]).resolve())
                self.assertEqual(sha256_file(bundle.checkpoint), asset["sha256"])
                self.assertEqual(bundle.checkpoint.name, f"{bundle.bundle_id}.pt")
                self.assertTrue(self.store.verify(bundle.bundle_id)["valid"])
                self.assertEqual(bundle.fold, -1)
                self.assertEqual(bundle.training["seed"], 2)
                self.assertEqual(bundle.training["checkpoint_policy"], "last_fixed_budget")
                self.assertEqual(bundle.training["validation_ids"], [])
                self.assertEqual(len(set(bundle.training["train_ids"])), 24)
                self.assertEqual(bundle.training["evaluation_ids"], [3, 7, 18, 19, 26])
                self.assertFalse(set(bundle.training["train_ids"]) &
                                 set(bundle.training["evaluation_ids"]))
                self.assertEqual(bundle.data["clip_len"], 15)
                self.assertEqual(bundle.data["num_points"], 1024)

    def test_inference_only_exports_and_provenance(self) -> None:
        protocol_hashes = set()
        for method in METHODS:
            with self.subTest(method=method):
                bundle = self.store.get(f"{method}_final24_len15_seed2")
                checkpoint = torch.load(bundle.checkpoint, map_location="cpu", weights_only=True)
                self.assertNotIn("optimizer", checkpoint)
                self.assertNotIn("scheduler", checkpoint)
                self.assertEqual(checkpoint["config"]["model"], bundle.model)
                self.assertEqual(checkpoint["config"]["data"], bundle.data)
                metadata = json.loads((bundle.bundle_dir / "provenance.json").read_text())
                self.assertTrue(metadata["state_dict_unchanged"])
                self.assertEqual(metadata["checkpoint"]["sha256"], bundle.checkpoint_sha256)
                self.assertEqual(len(metadata["source_checkpoint_sha256"]), 64)
                protocol_hashes.add(metadata["source_protocol_sha256"])
                serialized = json.dumps({k: v for k, v in checkpoint.items() if k != "model"})
                self.assertNotIn("/home/", serialized)
                if method == "lidargaitpp":
                    self.assertEqual(checkpoint["iteration"], 10000)
                    self.assertEqual(bundle.model["SeparateBNNecks"]["class_num"], 24)
                else:
                    self.assertEqual(checkpoint["epoch"], 50)
                    self.assertEqual(bundle.model["num_classes"], 29)
        self.assertEqual(len(protocol_hashes), 1)

    def test_legacy_fold_lookup_is_unchanged(self) -> None:
        for method, method_key in (("pointnet_tmax", "pc_v1"), ("mph_gait", "mph_gait"),
                                   ("lidargaitpp", "lidargaitpp")):
            with self.subTest(method=method):
                legacy = self.store.find_legacy(method_key, fold=0)
                self.assertEqual(legacy.bundle_id, f"{method}_fixed_special5_seed0_split0")
                current = self.store.get(f"{method}_final24_len15_seed2")
                self.assertNotEqual(current.checkpoint_sha256, legacy.checkpoint_sha256)

    def test_cpu_forward_has_finite_normalized_descriptor(self) -> None:
        generator = torch.Generator().manual_seed(2)
        points = torch.randn(15, 1024, 3, generator=generator)
        points *= torch.tensor([180.0, 450.0, 100.0])
        points[..., 2] += 2000.0
        for method in METHODS:
            with self.subTest(method=method):
                adapter = ModelAdapter(self.store.get(f"{method}_final24_len15_seed2"), device="cpu")
                descriptor = adapter.encode_tensor(points)
                dimension = 7936 if method == "lidargaitpp" else 256
                self.assertEqual(tuple(descriptor.shape), (1, dimension))
                self.assertTrue(bool(torch.isfinite(descriptor).all()))
                self.assertTrue(torch.allclose(descriptor.norm(dim=1), torch.ones(1), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
