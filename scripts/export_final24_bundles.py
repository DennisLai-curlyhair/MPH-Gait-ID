#!/usr/bin/env python3
"""Export deployment bundles from trusted, locally trained Final-24 checkpoints."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BUNDLES = ROOT / "mph_gait_id/model_bundles"
METHODS = {
    "pointnet_tmax": "PointNet-TMax",
    "mph_gait": "MPH-Gait",
    "lidargaitpp": "LidarGait++",
}
FIXED_IDS = [3, 7, 18, 19, 26]
TRAIN_IDS = [person for person in range(1, 30) if person not in FIXED_IDS]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(source_root: Path, run_id: str, seed: int, method: str, output: Path) -> dict:
    run = source_root / method / run_id / f"seed_{seed}"
    protocol_path = run / "protocol.json"
    protocol = load_json(protocol_path)
    if (protocol.get("name") != "final24_fixed5_eval_v1"
            or protocol.get("train_ids") != TRAIN_IDS
            or protocol.get("final_eval_ids") != FIXED_IDS
            or protocol.get("val_ids") != []
            or protocol.get("checkpoint_policy") != "last_fixed_budget"
            or protocol.get("selection_uses_fixed_special") is not False):
        raise ValueError(f"Unexpected Final-24 protocol: {protocol_path}")

    template = BUNDLES / f"{method}_fixed_special5_seed0_split0/bundle.yaml"
    bundle = copy.deepcopy(yaml.safe_load(template.read_text(encoding="utf-8")))
    training = {
        "fold": -1,
        "seed": seed,
        "protocol": protocol["name"],
        "checkpoint_policy": "last_fixed_budget",
        "seed_selection": "publisher_selected_deployment_seed",
        "train_ids": TRAIN_IDS,
        "evaluation_ids": FIXED_IDS,
        "validation_ids": [],
        "amp": False,
    }
    if method == "lidargaitpp":
        source_manifest = load_json(run / "manifest.json")
        if source_manifest["randomness"]["training_seed"] != seed:
            raise ValueError("LidarGait++ source seed does not match")
        if (source_manifest["clip_len"] != 15 or source_manifest["num_points"] != 1024
                or source_manifest["drop_first_frames"] != 30
                or source_manifest["coordinate_adapter"]["mapping"]
                != "[forward, lateral, height_up] = [Z, X, -Y] * unit_scale"):
            raise ValueError("LidarGait++ source input contract does not match")
        candidates = list(run.glob("train_work/output/**/checkpoints/*-10000.pt"))
        if len(candidates) != 1:
            raise ValueError("Expected exactly one final 10000-iteration checkpoint")
        source = candidates[0]
        config_path = run / "configs/train.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model = dict(config["model_cfg"])
        model["name"] = model.pop("model")
        if (model["SeparateBNNecks"]["class_num"] != 24
                or config["trainer_cfg"]["total_iter"] != 10000):
            raise ValueError("Unexpected LidarGait++ classifier or training budget")
        training["iteration"] = 10000
        provenance = load_json(run / "provenance.json")
        training["upstream_repository"] = "https://github.com/ShiqiYu/OpenGait.git"
        training["upstream_commit"] = provenance["official_commit"]
    else:
        source = run / "checkpoints/final.pt"
        config_path = run / "config.json"
        config = load_json(config_path)
        if config["experiment"]["seed"] != seed:
            raise ValueError("Point-model source seed does not match")
        for key in ("clip_len", "num_points", "drop_first_frames", "coordinate_adapter"):
            if config["data"][key] != bundle["data"][key]:
                raise ValueError(f"Point-model input contract differs: {key}")
        model = dict(config["model"])
        training["epoch"] = 50
        if method == "mph_gait":
            model.update(use_coarse_windows=True, use_shifted_windows=False,
                         use_token_transformer=True)

    from mph_gait_id.checkpoint_io import load_checkpoint
    original = load_checkpoint(source)
    step_key = "iteration" if method == "lidargaitpp" else "epoch"
    if original.get(step_key) != training[step_key]:
        raise ValueError(f"Checkpoint is not the final fixed-budget state: {source}")
    state = original["model"]
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint has no model state dictionary")
    for key, value in state.items():
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid model tensor: {key}")

    bundle_id = f"{method}_final24_len15_seed{seed}"
    target = output / bundle_id
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle: {target}")
    bundle.update(
        bundle_id=bundle_id,
        display_name=f"{METHODS[method]} Final-24 len15 seed{seed}",
        description=f"Final-24 deployment checkpoint; seed {seed}, final fixed training budget.",
        model=model,
        training=training,
    )
    source_record = {
        "source_run": f"{method}/{run_id}/seed_{seed}",
        "source_checkpoint": source.relative_to(run).as_posix(),
        "source_checkpoint_sha256": sha256(source),
        "source_config_sha256": sha256(config_path),
        "source_protocol_sha256": sha256(protocol_path),
        "exporter_sha256": sha256(Path(__file__)),
        "export_torch_version": str(torch.__version__),
        "export_format": "inference_only_v1",
        "state_dict_unchanged": True,
        "excluded": ["optimizer", "scheduler", "local_training_paths"],
    }
    # Only inference configuration is serialized; the original training run is unchanged.
    payload = {
        "model": {key: value.detach().cpu().clone() for key, value in state.items()},
        "config": {"model": model, "data": dict(bundle["data"])},
        step_key: training[step_key],
        "deployment": {**training, **source_record},
    }
    return {"target": target, "bundle": bundle, "payload": payload,
            "provenance": source_record, "original_state": state}


def export(plan: dict) -> None:
    target, bundle = plan["target"], plan["bundle"]
    target.mkdir(parents=True, exist_ok=False)
    checkpoint = target / f"{bundle['bundle_id']}.pt"
    torch.save(plan["payload"], checkpoint)
    reloaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = plan["original_state"]
    if (reloaded["model"].keys() != expected.keys()
            or any(reloaded["model"][key].dtype != value.dtype
                   or not torch.equal(reloaded["model"][key], value)
                   for key, value in expected.items())):
        raise RuntimeError("Export changed the model state")
    bundle["checkpoint"] = {"file": checkpoint.name, "sha256": sha256(checkpoint)}
    (target / "bundle.yaml").write_text(
        yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8")
    (target / "provenance.json").write_text(
        json.dumps({**plan["provenance"], "checkpoint": bundle["checkpoint"]}, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"bundle_id": bundle["bundle_id"], **bundle["checkpoint"],
                      "size_bytes": checkpoint.stat().st_size}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-id", default="20260817-182113_final24_len15_3seed")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--output-dir", type=Path, default=BUNDLES)
    args = parser.parse_args()
    plans = [prepare(args.source_root.resolve(), args.run_id, args.seed, method,
                     args.output_dir.resolve()) for method in METHODS]
    hashes = {plan["provenance"]["source_protocol_sha256"] for plan in plans}
    if len(hashes) != 1:
        raise ValueError("The three source protocol files differ")
    for plan in plans:
        export(plan)


if __name__ == "__main__":
    main()
