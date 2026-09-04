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
Gallery SQLite databases
outputs / live enrollment sessions / caches
virtual environments / IDE files / Python bytecode
SAM ViT-B optional checkpoint
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

專案授權：**TBD**

第三方模型與權重再散布狀態：**TBD**

詳見 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
