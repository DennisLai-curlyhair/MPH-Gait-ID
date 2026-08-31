# 即時效率測試操作守則

## 測試目的

第四頁用 Azure Kinect 即時比較完整處理鏈與點雲模型的效率：

```text
Kinect 讀取 -> 人體偵測 -> 人體點雲前處理 -> 模型抽特徵 -> Gallery 比對
```

測試只在記憶體中處理感測資料。報告只含時間、FPS、計數、模型設定與裝置資訊，不保存 RGB、深度影像、點雲陣列或 embedding。

## 建議的標準設定

- 人物偵測：YOLO person segmentation
- Device：`auto`，並確認實際使用 CUDA
- Clip length：使用 bundle 預設值，三個內建模型目前都是 15
- Inference stride：5
- 暖機：10 秒
- 正式測量：60 秒
- 受測狀態：靜坐（只測效率）
- 每個模型：至少三次

比較模型時，除了模型以外的所有條件都必須相同。尤其不要在 MPH-Gait ID 使用 YOLO-Seg、LidarGait++ 卻改用 SAM。

## 測試前

1. 關閉 Teams、Webex 或其他可能占用 Kinect 的程式。
2. 第二頁的即時辨識／註冊必須停止；第二頁和第四頁不能同時使用 Kinect。
3. 只留一位受測者在偵測範圍內。
4. 固定 Kinect 位置、椅子、受測距離、背景、照明與姿勢。
5. 若要納入 Gallery 比對成本，各模型的註冊人數及 embedding 數量應盡量一致。
6. 第一次測試建議先以 PointNet-TMax 跑 20 秒，確認畫面顯示有效點數且沒有持續等待人物。

可先執行 `python mph_gait_id/scripts/doctor.py --realtime --kinect`。此檢查會實際嘗試取得相機，不只是確認 USB 裝置數量。

## 標準操作

1. 進入第四頁「即時效率測試」。
2. 選擇模型，例如 MPH-Gait ID。
3. 選擇固定的偵測器、Device、stride、暖機秒數與測量秒數。
4. 靜坐測試選「靜坐（只測效率）」；`Expected Person ID` 可留空。
5. 按「開始效率測試」。模型載入和 Kinect 開啟時間不列入正式測量。
6. 保持完整人體在相機範圍內。系統取得第一個有效人體點雲後才開始暖機。
7. 暖機結束後不要移動相機、切換程式、開啟其他相機軟體或改變設定。
8. 正式測量結束後系統會自動停止並保存 JSON 報告。
9. 換下一個模型，保持相同條件重做。
10. MPH-Gait ID、LidarGait++ 各做至少三次，下一輪交換先後順序。

建議測試順序：

```text
第 1 輪：MPH-Gait ID -> LidarGait++
第 2 輪：LidarGait++ -> MPH-Gait ID
第 3 輪：MPH-Gait ID -> LidarGait++
```

這能降低模型暖機順序、人體疲勞與環境隨時間變化造成的偏差。

## 指標判讀

- `E2E FPS`：測量期間完整 pipeline 實際處理的 frame 數／秒。
- `Valid FPS`：成功產生有效人體點雲的 frame 數／秒；比瞬時 Pipeline FPS 更重要。
- `Model P50`：一般情況下的模型抽特徵時間。
- `Model P95`：較慢的 5% 推論情況；用來判斷偶發卡頓。
- `Gallery P50`：Gallery 載入與比對的一般耗時，會受特徵維度與 Gallery 大小影響。
- `Valid %`：成功產生人體點雲的幀比例。過低時，模型耗時比較可能被偵測失敗干擾。
- `Recognition updates/s`：每秒實際更新辨識結果的次數，通常受 Valid FPS 與 stride 共同影響。
- `gpu_memory.peak_allocated_mib`：正式測量期間 PyTorch 的 GPU 記憶體配置峰值；使用 CPU 時不提供。

比較時優先看三次測試的中位趨勢，不要只挑最高 FPS。若某一次 Valid % 明顯較低，應先檢查遮擋、多人入鏡或人體點數不足，再決定是否排除該次。

## 靜坐測試的限制

靜坐非常適合比較效率，因為人物位置、點數與背景較穩定，而且模型仍會收到固定大小的 `T x N` 點雲序列。

但靜坐沒有正常步態，因此下列結果不能由靜坐測試推論：

- 身分辨識正確率
- Unknown 門檻是否合理
- 行走中的穩定辨識時間
- 正面步態與遮擋下的辨識能力

若要評估以上項目，改選「行走（觀察結果行為）」並填入預期 Person ID，使用固定路線與多位受測者重複測試。

## 報告位置與匯出

每次有正式測量資料時自動保存：

```text
outputs/performance_benchmarks/benchmark_*.json
```

在結果表格選取一列後，可按「匯出選取報告」另存：

- JSON：保留完整階層與判讀註記。
- CSV：展開成單列，方便放入 Excel、pandas 或統計軟體。

手動按「停止並保留部分結果」時，已進入正式測量的數值會保存並標記為未完整完成；若仍在等待人物或暖機前就停止，則不建立空報告。
