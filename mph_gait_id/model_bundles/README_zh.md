# 點雲模型 bundles

目前內含三種點雲模型，各提供 Fixed-Special5 與 Final-24 兩種權重：

| Bundle | 架構 | 輸入 |
|---|---|---|
| `pointnet_tmax_fixed_special5_seed0_split0` | PointNet-TMax | T × N × XYZ |
| `mph_gait_fixed_special5_seed0_split0` | MPH-Gait | T × N × XYZ |
| `lidargaitpp_fixed_special5_seed0_split0` | LidarGait++ | T × N × XYZ |
| `pointnet_tmax_final24_len15_seed2` | PointNet-TMax Final-24 | T × N × XYZ |
| `mph_gait_final24_len15_seed2` | MPH-Gait Final-24 | T × N × XYZ |
| `lidargaitpp_final24_len15_seed2` | LidarGait++ Final-24 | T × N × XYZ |

每個 bundle 由 `bundle.yaml` 與其中指定的 `.pt` 權重組成。Final-24 檔名包含模型、`final24_len15_seed2`，不再使用通用的 `checkpoint.pt`。`bundle.yaml` 定義架構、Frame 長度、點數、標準化、checkpoint hash 及其他相容性資料。

Final-24 使用固定五人以外的 24 人訓練；PointNet-TMax / MPH-Gait 為第 50 epoch，LidarGait++ 為第 10,000 iteration。三者統一提供 seed 2 作為部署版本，不代表每種模型的所有指標都以 seed 2 最佳。匯出只移除 optimizer、scheduler 與本地路徑，模型張量保持不變。

在介面選擇 `Final-24 len15 seed2` 即可載入。原本的預設選項與舊權重保留不變。換權重後需重新註冊 Gallery，舊模型的特徵不能混用；舊資料仍保留。Final-24 不屬於原本某個測試 split，因此 metadata 使用 `fold: -1`。

詳細來源與檢查方式見 [Final-24 說明](../../docs/FINAL24_WEIGHTS.md)。

可執行下列指令驗證：

```powershell
python mph_gait_id/scripts/validate_model_bundles.py
```

不要只替換 `.pt` 檔案；權重與 bundle metadata 必須成對一致。
