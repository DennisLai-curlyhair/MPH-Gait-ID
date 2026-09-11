# 註冊前景點雲來源庫

本功能為 Conference MPH-Gait ID 的階段 B：保存註冊時的前景點雲，提供
回放、來源管理與重新編碼所需的資料。[階段 C](SOURCE_REGISTRATION_zh.md)
可將選取的來源片段，一次重新註冊至多個已安裝模型；
階段 A 的 `.mphgallery` 匯出包仍只包含身分特徵，不包含來源點雲。

## 操作流程

1. 進入 Azure Kinect 即時模式，選擇註冊並填寫人物 ID、姓名。
2. 開始 session 前，勾選「同意保存註冊前景點雲」。每次啟動程式預設不勾選；
   操作者須取得受測者同意並遵守資料保存政策。
3. 依原流程錄製 passes，進入「完成並檢查」。
4. 選擇要註冊的 passes，確認提交。只有選取的 passes 會進入來源庫。
   人工接受有移動方向警告的片段時，仍保留警告與 override 記錄。
5. 切至「註冊來源點雲」，選擇來源與 pass，即可播放、暫停、逐幀、拖動時間軸
   與縮放。這是保存 XYZ 的正面視圖回放，不是重新辨識，也不是 RGB 影片。

Review 關閉後可重新打開，也可繼續錄製；暫存會保留。同一 session 的來源
包含選取 passes 的有效幀，不受 Gallery 最多儲存幾個 embeddings 的設定限制。
倒數、無人物、捨棄與未選取的片段不會成為永久來源。辨識模式不會保存來源。

捨棄 pass 會刪除該 pass 暫存；放棄 session、正常取消錄製或退出程式會清除
未提交暫存。提交期間不能關閉程式，以免中斷來源與 Gallery 寫入。

## 保存的位置與內容

```text
人物偵測與背景濾除
  ├─ 選擇性背景保存：每幀不同點數的 N x 3 FP32 前景 XYZ
  └─ 原有模型路徑：1024 點抽樣 -> 座標轉換／正規化 -> 特徵提取

Review 選取片段 -> 檢查來源檔案 -> 同一 SQLite transaction 寫入特徵與來源索引
```

保存內容仍是相機座標：X 向右、Y 向下、Z 向前，單位 mm。尚未做固定點數
抽樣、逐幀置中、模型座標轉換或尺度正規化，保留人體尺度與相機方向上的平移。
點雲是可供不同模型重新處理的資料；embedding 仍不能跨模型、權重直接共用。

不保存 RGB、mask 圖、原始深度圖與含背景的完整點雲。前景濾除可能仍有誤差，
且丟棄的背景無法還原，因此未來不能靠這份來源重新做人物分割或相機投影校正。

預設位置由 Gallery 資料庫名稱決定：

```text
data/gallery.sqlite3
data/gallery.sqlite3_enrollment_sources/
  .writer.lock
  staging/<source-uuid>/        # 未提交暫存
  records/<source-uuid>/
    manifest.json
    frame_000000.npy
    ...
  trash/                       # 刪除操作的中斷復原暫存
```

`enrollment_sources` 索引表放在同一份 SQLite，儲存來源 UUID、人物的永久
transfer UID、拍攝時的 ID／姓名、片段數、幀數、容量與 manifest SHA256。
來源 manifest 與每幀點雲都以 hash 核對；NPY 使用 `allow_pickle=False` 載入。

Manifest 記錄座標單位、儲存階段、sensor frame index、來源時間戳、取樣中斷、
pass 方向／品質、人工確認結果、原始註冊視窗與模型／權重資訊、偵測設定、
背景濾除參數及前處理程式 SHA256。時間戳沿用來源的時鐘，不假定為 UTC；
session 建立時間另外以 UTC 記錄。不能跨 pass 或取樣中斷串接成連續序列。

Azure 使用上游 SDK 的 color-aligned XYZ，不代表此功能有封存完整相機
calibration object。Replay 只有在來源 adapter 提供投影參數時才會記錄參數。

## 容量與異常處理

可在 `mph_gait_id/configs/system.yaml` 的 `realtime.foreground_storage` 調整：

| 項目 | 預設 |
|---|---:|
| 單一 session 上限 | 2 GiB |
| 來源庫容量上限 | 20 GiB |
| 磁碟最低保留空間 | 512 MiB |
| 背景待寫入 queue | 8 幀 |
| 每幀最大點數 | 2,000,000 |
| 每個 session 最大幀數 | 9,000 |

背景 writer 先複製資料，避免 SDK 重用影像緩衝區導致保存內容被改寫。若磁碟
過慢導致 queue 滿、容量超標或寫入失敗，會明確中止該次來源保存與註冊，
不會暗中漏幀後仍表示成功。既有 Gallery 不受影響。可改善磁碟效能，或重新
開始一個不保存來源的 session；保存點雲會增加 I/O，不能視為零成本。

來源檔案先準備完成，再將來源索引與 embeddings 放進同一筆 SQLite transaction。
交易失敗會回滾 Gallery 並刪除未發布的來源副本；Review 暫存仍可重試。提交
成功後若報告或暫存清理失敗，只會回報 warning，不應再次註冊相同片段。

OS 檔案鎖限制同一來源庫只有一個寫入者；錄製與 Review 期間不能從另一份程式
刪除或清理來源。異常斷電／關閉後，確認沒有作用中的 session，再按「清理未
提交暫存」。它會清除未提交的來源，並復原 SQLite 尚未完成刪除的來源目錄。
已提交但檔案損壞／缺失的來源會顯示錯誤，不會當作正常錄影使用。

## 刪除、備份與隱私

- 刪除來源：永久刪除該份 XYZ，不刪 Gallery 特徵；embedding 無法還原點雲。
- 刪除人物／片段特徵：來源保留。撤回同意時，需要另外刪除來源，並檢查
  session JSON、Gallery 舊備份及外部備份的保存政策。
- 改名後列表顯示目前姓名；manifest 保留拍攝時姓名。ID 後面的 `*` 表示來源
  沒有連結到目前 Gallery 人物。相同可見 ID 被另一人重用，不會承接舊來源。
- `.mphgallery` 匯出與 Gallery 的 SQLite 備份都不含 NPY。跨設備搬移來源請用
  [階段 D 來源移轉](SOURCE_TRANSFER_zh.md)，再透過[階段 C](SOURCE_REGISTRATION_zh.md)
  為已安裝的模型建立特徵。管理員完整備份時，先停止程式，再一併備份資料庫及
  對應來源目錄；只有 SQLite 無法回放來源。
- 資料庫、來源、暫存皆在 Git 忽略規則內，未將人物資料加入專案。資料並未
  加密；請使用適當的 OS 權限、磁碟／備份加密及保存期限。SHA256 是完整性
  核對，並非加密或抵禦資料庫持有人竄改的數位簽章。

## 本機驗收

```bash
python -m unittest discover -s mph_gait_id/tests
python -m unittest discover -s scripts/tests
python scripts/validate_release.py
python run.py ui
```

以實際 Kinect 檢查：未勾選時不保存、勾選後多 pass 錄製、捨棄／繼續錄製、
選取部分片段提交、來源回放、重新啟動後仍可查看、刪來源不影響特徵，以及
中英文與不同桌面縮放設定。自動測試使用暫存資料庫與合成點雲，不能取代
Windows 檔案系統、相機 SDK、實體 Kinect 與 GUI 驗收。
