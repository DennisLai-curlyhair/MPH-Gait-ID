# 點雲模型 bundles

目前開發版內含三個點雲模型：

| Bundle | 架構 | 輸入 |
|---|---|---|
| `pointnet_tmax_fixed_special5_seed0_split0` | PointNet-TMax | T × N × XYZ |
| `mph_gait_fixed_special5_seed0_split0` | MPH-Gait ID | T × N × XYZ |
| `lidargaitpp_fixed_special5_seed0_split0` | LidarGait++ | T × N × XYZ |

每個 bundle 由 `bundle.yaml` 與 `checkpoint.pt` 組成。`bundle.yaml` 定義架構、Frame 長度、點數、標準化、checkpoint hash 及其他相容性資料。

可執行下列指令驗證：

```powershell
python mph_gait_id/scripts/validate_model_bundles.py
```

不要只替換 `checkpoint.pt`；權重與 bundle metadata 必須成對一致。
