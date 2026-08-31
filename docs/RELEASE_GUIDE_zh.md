# GitHub／公開發布指南

## 建議提交

```text
Python source
configs/system.yaml
model bundle YAML
tests and scripts
asset manifest and SHA256
README / design / validation documents
```

## 不應提交

`.gitignore` 已排除：

```text
*.pt / *.pth checkpoints
Gallery SQLite databases
outputs / live enrollment sessions / caches
virtual environments / IDE files / Python bytecode
```

大型模型建議放 GitHub Release、Hugging Face 或機構檔案服務，並保留
`assets/manifest.yaml` 的相對 target、release name 與 SHA256。使用者可設定：

```text
GAIT_IDENTITY_ASSET_BASE_URL=<asset directory URL>
```

再執行 `scripts/download_assets.py --profile realtime-yolo`。

## 發布前檢查

```bash
python -m mph_gait_id.ui_app --check-only
python mph_gait_id/scripts/doctor.py
python mph_gait_id/scripts/download_assets.py --profile all --check
python -m pytest mph_gait_id/tests
```

另應依 `README.md` 的硬體驗證步驟，在乾淨 Windows 主機完成 Kinect 驗收。不要將
replay 通過寫成 Azure Kinect hardware validated。

## 授權

公開前必須決定本專案程式碼與資料集授權，並逐一確認 OpenGait／LidarGait++、
Ultralytics YOLO、Meta SAM、Azure Kinect SDK 與 checkpoints 的再散布條款。
在授權盤點完成前，不應直接把所有第三方程式碼與權重推到公開 repository。

詳見 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
