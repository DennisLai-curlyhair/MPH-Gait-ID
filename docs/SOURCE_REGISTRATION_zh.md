# 階段 C：來源點雲的多模型特徵註冊

本功能延續階段 B 保存的註冊前景點雲，一次為多個指定模型／權重建立各自的
Gallery embeddings。不需要重新走路，也不是重新訓練模型，更不是把 A 模型的
embedding 轉成 B 模型的 embedding。

## 操作流程

1. 升級測試前先備份 Gallery。停止辨識、註冊、效能測試，並完成或放棄待處理的 Review。
2. 開啟「註冊來源」，選一筆來源；確認人物、片段與點雲回放正確。
3. 按「多模型特徵註冊」，勾選要使用的片段與模型／權重。
4. 每個模型可各自設定 T，例如 PointNet-TMax=15、MPH-Gait=15、LidarGait++=30。
   預設使用 bundle 的 clip length；N 為 bundle 指定點數，不在這裡任意修改。
5. 設定「每片段最多特徵數」（預設 10）與運算裝置（預設 auto）。
6. 按「開始多模型註冊」並確認。模型依序載入與推論，視窗顯示逐模型、逐視窗進度。
7. 完成後，至 Gallery 或即時辨識頁選擇對應的模型／權重與相同 T，即可使用新特徵。

例如第一次用 MPH-Gait 註冊 P007 並同意保存前景點雲，之後可在來源頁同時勾選
PointNet-TMax 與 LidarGait++，一個工作完成兩套 Gallery 註冊，不必分別挑檔案。
若也勾選原本 MPH-Gait 的相同權重與 T，已註冊的片段會跳過，不重複增加特徵。

目前一次選一筆來源記錄；這筆來源可以含多個已通過 Review 的 pass，並同時指定
多個模型。不同來源記錄分開提交。只使用已安裝的點雲模型，不自動下載權重。

## 資料流程與公平性

```text
階段 B 前景 XYZ：可變點數、camera 座標、mm
  -> 驗證 manifest／使用幀的 SHA256 與人物 UID
  -> 每個 pass 依取樣中斷切成連續段
  -> 各模型依自己的 T 建立完整視窗
  -> 使用既有即時 deterministic sampler 抽成該模型的 N 點
  -> 既有 ModelAdapter：座標轉換、模型所需正規化
  -> 指定 checkpoint -> L2-normalized embedding
  -> 所有新結果在同一 SQLite transaction 提交
```

不修改原始來源、模型架構、權重或現有即時推論路徑。LidarGait++ 仍使用既有
normalized XYZ 加 metric height 的輸入處理，descriptor 維度由 part head 判讀。
各模型的相似度空間彼此獨立，絕不共用 embeddings。

來源的 preprocessing profile 必須與目前應用程式相同；不會將舊前景冒稱為經過
新版背景濾除的資料。模型權重、T、N 等依原有相容性紀錄隔離保存。

**不再丟棄前 30 幀。** 這裡已是註冊時保存的有效行走資料，不是 benchmark 的
完整原始影片。為相容現有即時 Gallery，model record 仍保留 bundle 原有的
`drop_first_frames` 欄位，但不會據此再裁切這些保存幀。

視窗 stride=T，無重疊；尾端不足 T 幀直接捨棄，不補幀、不重複幀。不跨 pass、
segment break、逆向時間或超過保存 gap 門檻的取樣中斷。若完整視窗超過上限，
從所有合格視窗等距選取。上限是「每模型、每 pass」，不是整個人物的終身上限。

例如一個沒有中斷的 120-frame pass：T=15 可得 8 個 embedding，T=30 可得 4 個。
同一份來源在不同模型／T 下的 embedding 數量不一定相同，這是正常行為。

## 結果狀態

| 狀態 | 含意 |
|---|---|
| already_registered | 該來源／pass 在相同 model key 已有特徵，包含停用資料，整個 pass 跳過 |
| too_short | 取樣中斷後沒有足夠的連續 T 幀；其他有效片段仍可註冊 |
| registered | 該模型的新特徵已成功提交 |
| skipped | 沒有尚待註冊的合格 pass，不載入該模型 |
| failed / cancelled | 本次工作不增加任何模型的 Gallery 特徵 |

停用的特徵不會因重新編碼而自動啟用，請從 Gallery 管理操作。
若某 pass 的特徵已全部永久刪除，可在明確確認後從保留來源重新產生新特徵；
這是新的編碼／註冊，不是還原 archive 的 tombstone。
若 pass 仍有部分特徵存在，會保守地跳過整個 pass，不自動補齊個別視窗。

來源人物若已刪除或停用，先透過 Gallery 管理或階段 A 匯入還原原人物。
目前不提供自動轉移 orphan source 的人物歸屬。同一個 Person ID 被新人物使用，
不代表擁有舊人物的來源點雲，系統會拒絕直接掛上去。

## 取消、失敗與資料保護

- 「取消工作」會等待正在執行的那次推論返回，再取消後續流程，不強制中斷 CUDA。
- 各模型先編碼完成，最後才一次寫入；後面模型失敗、NaN、權重損毀或資料庫寫入失敗，
  不會留下只完成第一個模型的半套註冊。
- 原本已存在的 Gallery 與來源檔案不受取消／失敗影響。
- 提交前再次確認人物 UID、bundle／checkpoint 與重複來源，避免背景計算期間資料被改動。
- 執行期間鎖住來源庫，避免另一個 app instance 刪除、清理或開始來源錄製。
  同一 UI 也會阻擋競爭的辨識、註冊、Gallery 管理與效能測試。
- 一次只載入一個新目標模型，不把這些模型加入持久 runtime cache。
  系統不能控制其他外部程式的 GPU 使用量。

完成／失敗／取消的摘要保存在同一 Gallery SQLite 的 `enrollment_source_jobs`，
包含來源 ID/hash、人物 UID、目標權重 hash、T/N、pass 狀態與新增筆數。
資料庫本身不可寫時，失敗紀錄只能盡力保存；強制關閉程式也可能來不及記錄失敗，
但 SQLite transaction 不會留下部分模型寫入的狀態。

階段 A `.mphgallery` 仍只匯出特徵，不含前景 NPY 或本機工作紀錄。
資料庫與來源都屬生物特徵資料，不應上傳 GitHub。不同模型的 unknown threshold
仍需分別驗證，不會隨此次來源重新註冊而自動取得校準值。

## 驗證與 Local 測試

在專案根目錄執行：

```bash
python -m unittest discover -s mph_gait_id/tests
python scripts/smoke_source_registration.py
```

Smoke test 使用暫存的合成前景點雲，實際載入三個 Final-24 checkpoint，以 CPU
比對來源重新編碼和相同即時視窗的輸出，並檢查 Gallery 匯出／匯入與重複跳過。
不連 Kinect、不下載權重，也不在正式 Gallery 新增測試人物。

Local 驗收清單：

- [ ] 同一個來源同時註冊到 PointNet-TMax、MPH-Gait、LidarGait++。
- [ ] 各模型與相同 T 可在 Gallery 看到該人物並執行辨識。
- [ ] 再按一次，已註冊 pass 不增加筆數。
- [ ] 中途取消，所有模型的 Gallery 筆數維持原狀。
- [ ] T=15 / T=30、太短片段、取樣中斷均按上述規則處理。
- [ ] 原本回放、來源刪除、Gallery 管理、特徵匯出／匯入不受影響。
- [ ] 中／英文與本機螢幕縮放下，選取欄位、捲動與取消按鈕正常。

本版與階段 B 共用 `conference/enrollment-source-library` 分支，可一起測試、PR、merge。
尚未進行 Kinect 實機及本機桌面互動驗收，不將合成資料測試視為相機實測。
