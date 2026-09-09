# Gallery 匯出與匯入

本功能屬於 Conference 應用程式的階段 A：將已註冊的身分特徵搬移至另一台
電腦，不重新訓練、不重新拍攝，也不將不同模型的 embedding 互相轉換。

## 操作方式

1. 先停止離線作業、即時辨識／註冊和效能測試。
2. A 電腦開啟「Gallery 管理」，按「匯出全部 Gallery」，產生 `.mphgallery`。
   匯出包含所有模型與人物，不受目前頁面篩選條件限制，停用資料也會保留。
3. 將匯出檔透過可信且受保護的管道搬到 B 電腦；另外安裝相同模型及權重。
4. B 電腦按「匯入 Gallery」，在「人物」與「模型」頁籤檢查預覽。
5. 新人物預設新增；既有 ID 一律視為衝突，即使姓名相同也不自動合併。
   衝突預設略過，可「另建人物 ID」或「合併至既有人物」。合併前必須確認
   為同一個人，並保留 B 電腦原有的姓名、備註和人物狀態。
6. 按「確認匯入」。完成後會顯示新增、重複、略過特徵數，以及備份和保留包的位置。
7. 在正常辨識頁選擇相符的模型與 T，先用已知測試片段確認結果。

已匯入過的人物會維持原有 ID 對應。重複匯入不增加相同 embedding，也不會
重新啟用 B 電腦已停用的資料。此功能是合併匯入，不是 A/B 雙向同步；A 電腦
後來的改名、停用或刪除，不會自動覆寫 B 電腦的決定。

## 模型相容性

系統核對 checkpoint SHA256、架構／模型設定、輸入表示、點數、座標轉換、
正規化、T、丟棄開頭 frame 數、前處理 profile 及編碼程式指紋。
同樣 256 維不代表可混用；不同 checkpoint 也不能直接共用 embedding。
路徑與顯示名稱不作為特徵相容依據，相同設定但不同名稱的 bundle 可對應至本機。
程式指紋統一換行與路徑分隔符，避免 Windows／Linux 的正常差異造成不相容。

匯入不載入或執行模型權重，只檢查本機權重檔案的雜湊。原本的模型識別 key
不變，不會因新增匯入功能就讓舊 Gallery 消失。
前處理檢查使用紀錄中的 profile 名稱，不能證明同一 profile 下歷來的偵測／
濾除參數與相機校正皆相同，這些項目仍需在目的設備上另行確認。

舊註冊紀錄未完整保存註冊當時的所有編碼設定。第一次匯出會以本機當下的
bundle 與編碼程式補充比對資訊，請確認它們仍與原本註冊時一致。
匯出工具不能回溯證明以前沒有修改過設定。從匯入包取得的設定則會保留於
後續匯出，不因改版重新宣稱成新設定。

找不到相容模型的特徵不會混入現用 Gallery，而是保留於本機匯入包。
安裝相符 bundle 後，可以再次匯入該檔案；已完成的資料會略過。
更換 checkpoint／編碼方法時仍需重新註冊。只有 embedding 的舊即時紀錄
無法還原原始點雲；保存點雲及跨模型重建屬於後續階段 B/C。

## 資料內容與保護

匯出包只包含版本化 JSON 人物／模型／來源紀錄，以及數值型 FP32 embedding。
不包含權重、點雲、RGB/depth、完整 session 檔案、A 電腦的來源絕對路徑、
相機序號、密碼或辨識 threshold。姓名與備註會保留，請自行檢查敏感文字。
來源改以可攜式識別碼表示，不代表 B 電腦能播放原始影片。

匯入會檢查 SHA256、檔案大小、版本、關聯紀錄、向量維度、有限值與 L2 norm。
不解壓縮到任意路徑，不反序列化 pickle／PyTorch。大小上限為 512 MiB，
每個 JSON 上限 16 MiB，每種紀錄上限 100,000 筆。

匯出使用同一個 SQLite transaction snapshot。匯入前會先保留已驗證的包，
並建立一致性 SQLite 備份，再以單一 transaction 合併資料。
預覽後 Gallery、模型或匯入包若改變，必須重新預覽；合併失敗會回滾 Gallery。
已保留的匯入包和備份可能繼續存在，供排錯與重試。

以下路徑位於「目前設定的資料庫」旁，而不是 A 電腦的路徑：

```text
data/gallery_transfer/incoming/<sha256>.mphgallery
data/gallery_transfer/backups/gallery-<unique-id>.sqlite3
```

如需回復備份，先完整關閉應用程式及所有使用資料庫的程序。另存目前資料庫
及其 sidecar 檔案，再將備份還原至設定的資料庫位置。不要覆蓋正在使用的
資料庫；本版本不會自動執行破壞性的回復操作。

**匯出檔、保留包與備份包含敏感身分特徵，尚未加密。** 請使用受保護的設備、
加密儲存／傳輸管道並訂定保留期限。SHA256 只能驗證完整性，不能證明寄件者可信。
`.mphgallery`、`gallery_transfer/` 與資料庫備份已加入 Git 忽略規則，
但忽略規則不是權限保護，也不會刪除曾經提交的檔案。

換電腦後仍要檢查代表性的 probe；不同硬體／套件不保證逐 bit 相同的推論。
換攝影機或場地可能影響準確率和相似分數，threshold 需另行確認，不隨匯入而認定有效。

## 指令與驗證

在專案根目錄與已安裝的應用環境執行：

```bash
python -m mph_gait_id.gallery_transfer_cli export gallery.mphgallery
python -m mph_gait_id.gallery_transfer_cli preview gallery.mphgallery > data/import_preview.json
python -m mph_gait_id.gallery_transfer_cli import gallery.mphgallery --preview-file data/import_preview.json --apply
python -m unittest discover -s mph_gait_id/tests -v
```

指定獨立資料庫時，將 `--database /path/to/gallery.sqlite3` 放在子命令之前。
匯入預設略過衝突 ID。需要手動對應時，用 `--person-map data/person_map.json`，
以每位預覽人物的 `uid` 為 key、目的 ID 為值；`null` 表示略過。
對應既有人物 ID 等同明確合併，務必確認身分。

自動測試使用暫存資料庫與合成特徵，涵蓋往返搬移、停用／去重、相容性、
衝突、損壞檔案、過期預覽與回滾。這不等於 Azure Kinect 實機或跨設備準確率驗證。
