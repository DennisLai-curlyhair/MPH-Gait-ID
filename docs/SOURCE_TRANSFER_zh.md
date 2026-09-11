# 來源點雲匯出與匯入（階段 D）

階段 D 將階段 B 保存的註冊前景點雲搬到其他電腦，並保留階段 C 重新編碼所需的
資訊。它與[Gallery 特徵移轉](GALLERY_TRANSFER_zh.md)分開，不需要重新拍攝人物；
匯出／匯入本身不需相機、GPU、模型推論或新增套件。

## 兩種資料包的區別

| 項目 | `.mphsources` 來源包 | `.mphgallery` 特徵包 |
|---|---|---|
| 前景 XYZ 點雲 | 選取的已提交來源 | 不含 |
| 逐幀時間、點數、片段與中斷邊界 | 保留 | 僅特徵視窗資訊 |
| 座標、單位與前處理資訊 | 保留 | 模型編碼相容性資訊 |
| 人物可攜式識別碼、ID、姓名 | 保留 | 保留 |
| 模型專屬 embedding | 不含 | 保留 |
| 權重、RGB、原始 depth、全場景點雲 | 不含 | 不含 |

**來源點雲不等於 embedding，也不是完整原始場景。** 它是已濾背景、尚未固定點數
抽樣的 `N x 3 float32`，使用 camera XYZ（X 向右、Y 向下、Z 向前），單位 mm，
尚未中心化或轉軸，保留尺度與平移。無法從這份資料還原被濾除的背景或重新執行分割。

匯出單位為一筆已提交的來源錄影，包含該來源 review 選取 passes 的所有已保存幀，
不是只有產生 embedding 的視窗。本版不提供指定部分幀／pass 的匯出。

## 電腦 A 到 B 的操作

1. 結束或放棄未完成的註冊 review，停止其他作業。
2. A 開啟「註冊來源點雲」，選一筆，或按 Ctrl／Shift 多選來源。
   多選時預覽第一筆；刪除及多模型註冊仍須選取單筆。
3. 按「匯出選取來源」，指定來源庫外的 `.mphsources` 路徑並確認個資提醒。
   完成後顯示筆數、幀數、位置與 SHA256。失敗或取消不會覆寫原本的同名匯出檔。
4. 透過受保護的方式搬到 B。**資料包沒有加密，勿上傳公開 GitHub。**
5. B 按「匯入來源點雲」，選資料包，等待完整驗證。「人物」與「來源片段」分頁
   顯示人物對應、來源 UUID、片段／幀數、容量、前處理版本與重複／衝突狀態。
6. 處理人物 ID 衝突，按「確認匯入」。僅預覽不會建立人物、來源或 embeddings。
7. 關閉視窗後可直接回放；若要建立其他模型的 Gallery，安裝對應 bundle／權重，
   選來源後執行[多模型特徵註冊](SOURCE_REGISTRATION_zh.md)。

如果舊 embeddings 也需要保留，另外移轉 `.mphgallery`。**兩種包可依任意順序
匯入，但人物對應要一致。** 相同 portable UID 會連結到同一人，不依絕對路徑判定。
既有特徵仍需相符的模型、權重與前處理契約才能辨識；來源回放不需要模型權重。
threshold、UI 設定與自訂 bundles 不在本資料包中。

來源 UUID、session ID、pass ID 保留，因此階段 C 能跳過已存在相容特徵的片段，
停用的特徵亦納入重複檢查。選不同模型會建立各自的特徵，不是把一個 embedding
直接給另一個模型使用。

## 人物對應與重複處理

| 狀態 | 處理 |
|---|---|
| 新人物 | 未使用的 ID 預設納入；可另建 ID。 |
| 已對應人物 | 沿用既有 portable UID 對應，不能改接另一人。 |
| 相同 ID、不同 UID | 預設略過；另建未使用的 ID，或明確確認「合併至」同一人物。 |
| 人物曾被永久刪除 | 預設略過；勾選還原後使用未占用 ID。被別人重用的 ID 不會自動取回。 |
| 相同來源 UUID 與可攜 manifest | 驗證本機幀檔案後略過，不再複製點雲。 |
| 同 UUID、不同 manifest，或不同 UUID、相同內容指紋 | 阻止該人物匯入，需檢查來源，不可改檔名繞過。 |

「納入人物」可恢復符合條件的略過項目；已刪 ID 若被占用，使用「另建 ID」。
人物操作會套用到該人物在包內的所有來源。明確合併不覆寫目的端現有姓名、備註及
啟用狀態，也不刪除或重新啟用 Gallery 特徵。

**匯出後刪除來源，再匯入，可以還原點雲。** 若連人物也刪除，則必須明確還原人物。
原人物已刪除的孤立來源暫不允許匯出，需先還原原人物；重建同名 ID 不是還原。
這是資料移轉與還原，不是跨電腦同步刪除或同步改名。

## 保存內容與位置

```text
enrollment-sources.mphsources
  manifest.json                  # mph-source-transfer-v1
  sources/<source-uuid>/frame_000000.npy
  sources/<source-uuid>/...

data/
  gallery.sqlite3
  gallery.sqlite3_enrollment_sources/
    staging/transfer-<uuid>/      # 匯入暫存，不出現在來源清單
    records/<source-uuid>/
      manifest.json              # mph-foreground-source-v1
      frame_000000.npy
  source_transfer_backups/gallery-<unique-id>.sqlite3
```

可攜 manifest 保留人物 UID、匯出當下的 ID／姓名／啟用狀態、拍攝時 ID／姓名、
建立時間、來源及 session/pass ID、來源時鐘 timestamp、幀順序／中斷邊界、NPY
大小與 SHA256、座標／單位／儲存階段、已有的方向／品質摘要、錄製 opt-in 紀錄，
以及白名單內的偵測器設定、filter parameters、preprocessing profile 與 source hash。

不包含本機路徑、設備序號、模型設定、原本資料庫的 embedding ID、任意 metadata、
逐視窗編碼結果或完整相機校正／review 紀錄。錄製同意紀錄不等於已獲得分享授權。
NPY bytes 原樣保存；移除非必要 metadata 後，匯入的 manifest 有自己的 SHA256，
可能不同於原始 manifest。來源身分靠 UUID，不靠資料夾路徑或 manifest hash。

資料庫新增 `enrollment_source_transfers`，記錄已提交匯入的時間、archive SHA256、
筆數、來源 ID、還原人物數與備份位置。不重寫原本的來源或 embeddings，也不變更
模型編碼契約。資料包不另外複製成永久本機副本，需要備份時請保留匯出檔。

## 完整性、容量與復原

- 預覽驗證所有 frame 的 SHA256、檔案大小、受限 NPY header、shape、float32
  與有限數值。拒絕 object/pickle、不安全路徑、symlink、重複 ZIP member／JSON
  key、額外檔案或不支援的版本；不直接呼叫不受限的 ZIP 解壓縮。
- 匯入重新核對整包 SHA256 及人物／來源資料庫快照。預覽後變動需重新預覽。
- 每包最多 1,000 筆來源、100,001 個 ZIP members、64 MiB manifest、壓縮及
  解壓總量各 32 GiB，每幀最多 2,000,000 點；大量資料分批打包，逐幀讀寫。
- UI 沿用 `mph_gait_id/configs/system.yaml` 的
  `realtime.foreground_storage.library_limit_bytes`（預設 20 GiB）與
  `min_free_bytes`（預設 512 MiB）。匯入保留新來源及 SQLite 備份空間；
  匯出依未壓縮大小加上額外空間保守檢查。
- 共用來源庫 writer lease，避免錄製、刪除、清理、重新編碼及移轉同時寫入。
  處理中按取消會等待背景清理完成，不直接關閉正在寫入的執行緒。
- 提交前失敗／取消會 rollback 並清除未發布檔案；提交後才到達的取消不會撤銷
  已完成匯入。斷電後關閉其他程式，再使用「清理未提交暫存」清除孤立 staging
  或未提交檔案，不要手動混合不同版本的 SQLite。
- 既有來源若缺檔／損壞會報錯，不當作正常重複略過。確認備份後可透過 UI 刪除
  損壞來源，再從驗證通過的包還原。
- 匯入前 SQLite 備份不含 NPY，不能單獨當成完整系統備份。完整管理員備份需
  停止應用程式，一起保存資料庫、對應來源庫與自訂 bundles。

**SHA256 是完整性核對，不是加密、身分驗證或數位簽章。** 只匯入可信來源，遵守
分享授權與保存期限；資料包與 SQLite 備份都包含身分資料。
`.mphsources`、來源庫及備份已加入 Git 忽略規則，仍不可手動強制上傳公開。

## 驗證指令與 Local 測試

```bash
python -m unittest mph_gait_id.tests.test_source_transfer mph_gait_id.tests.test_source_transfer_ui -v
python -m unittest discover -s mph_gait_id/tests
python scripts/smoke_source_registration.py --device cpu --source-transfer
```

Smoke test 使用合成點雲及內附 PointNet-TMax、MPH-Gait、LidarGait++ Final-24
權重，核對移轉後點雲一致、與 live-window 編碼一致，以及 Gallery／來源移轉後
的重複註冊保護。不使用真人資料，不代表辨識準確率測試。

合併前請在接收端 Windows 測試：多來源匯出、中英文與桌面縮放、空資料庫匯入、
回放、多模型註冊、重複匯入、同 ID 不同人、刪來源後還原、已刪人物明確還原、
大包取消及重新啟動後仍可使用。伺服器測試不能取代本機 GUI／檔案系統驗收。
