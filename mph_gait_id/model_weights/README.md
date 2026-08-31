# Detector Weights

The repository includes `yolov8n.pt` and `yolov8n-seg.pt` for the default
single-person foreground pipeline. Their SHA256 values are recorded in
`../assets/manifest.yaml`.

The optional SAM ViT-B checkpoint is not committed because it is approximately
375 MB. Download and verify it with:

```bash
python mph_gait_id/scripts/download_assets.py --profile sam
```

Third-party model licenses and notices remain applicable.
