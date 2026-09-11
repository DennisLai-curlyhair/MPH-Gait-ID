# 即時點雲步態身分辨識系統設計

## 邊界

系統只處理人體前景點雲。支援的模型架構為 `pc_v1`、`mph_gait` 與 `lidargaitpp`；模型輸入均來自固定長度的 XYZ 點雲 window。

即時流程由裝置、人物分割、點雲前處理、window 取樣、模型推論與 Gallery 比對組成。RGB 畫面只供人物偵測與操作回饋，不會成為步態模型輸入。

## 相容性鍵

Gallery 特徵至少依下列欄位隔離：

- model bundle
- Frame 長度
- 捨棄開頭幀數
- 點雲處理版本

UI 將舊稱 profile 改成「點雲處理版本」。內部資料庫仍保留 `preprocessing_profile_id` 欄位以維持 schema 相容，但不再將這個技術名稱呈現給使用者。

## 人物前景

YOLO detection 以人物框選擇主要人物；YOLO-Seg 或 SAM 可取得更緊密的人物遮罩。遮罩透過 Kinect 校正資訊對齊 XYZ，接著套用深度範圍與點雲清理。註冊預設允許多人但只使用主要人物，UI 可切換成嚴格單人模式；辨識模式同樣使用主要人物策略。

最終深度過濾 mask 可映射回 RGB 作為綠色預覽。這條顯示路徑不參與 window、模型特徵或 Gallery 寫入，也不是步態影像模型。

## 時間與 window

裝置與 replay 都使用 monotonic timestamp。window metadata 保存實際首尾 frame、有效 sampling FPS、平均 gap 與最大 gap。時間倒退、零 gap 或超過設定上限時會清空序列，避免把不連續畫面組成一個步態片段。

## 引導式註冊

每個 pass 獨立記錄資料類型、觀察到的移動方向、位移及運動單調性。由於現有模型只訓練正面資料，UI 只提供正面／面向相機的 capture type。軌跡分類只能判斷平移，不能判斷身體正面或背面，因此它是警告而非硬性阻擋；有 embedding 的 pass 仍可由操作者人工納入。

目前移動方向判定使用每幀人體前景點雲的中位數 X、Z 中心。起點與終點分別取軌跡前後約 20%（至少 2、最多 10 幀）的中位數以降低抖動；Z 減少至少約 250 mm 判成靠近相機，Z 增加判成遠離相機，X 位移至少約 150 mm 則判成左右移動。沿主要軸至少 65% 的相鄰步進必須同向（容許 10 mm 抖動），否則標成靜止或疑似轉身。正面 capture 預期觀察到靠近相機，但警告可由操作者在提交前人工覆核。

註冊人物、模型與所有來源由同一 SQLite transaction 寫入。任一來源失敗會整批 rollback，避免人物存在但特徵不完整。

## 註冊來源點雲

階段 B 在固定點數抽樣前分流，透過 bounded queue 保存選擇性錄製的前景 XYZ。
Review 選取的來源索引與 embeddings 共用 SQLite transaction；檔案先準備，
交易失敗時移除未發布副本。來源人物使用永久 transfer UID，避免 ID 重用時
誤接舊資料。來源刪除與特徵刪除分離，也不包含於階段 A 匯出包。
詳見[註冊來源庫](ENROLLMENT_SOURCES_zh.md)。

## 跨模型來源註冊

階段 C 使用獨立的 `SourceRegistrationService` 與背景多模型註冊視窗，
把來源的連續 passes 依各目標 T 切成完整視窗，沿用既有 sampler 與 ModelAdapter。
每個模型依序載入，沒有 embedding 跨模型轉換；新特徵在確認人物、相容性與重複
資料後一次提交，取消或失敗不留下部分 Gallery。既有模型編碼檔案及階段 A
使用的 encoder hash 不變。詳見[多模型特徵註冊](SOURCE_REGISTRATION_zh.md)。

## Gallery

### 來源點雲移轉

階段 D 由 `source_transfer.py` 的 `SourceTransfer` 與 `ui/source_transfer.py`
的背景工作視窗負責。`.mphsources` 只包含已提交來源的白名單 manifest 與數值型
NPY，不載入模型；來源 UUID、人物 transfer UID、session/pass ID 與時間邊界皆保留。
預覽驗證封包後，匯入重新檢查封包／資料庫快照，先將點雲寫到 staging 並備份
SQLite，再以單一 transaction 提交人物對應及來源索引。沿用 writer lease 阻止
錄製、刪除、清理與編碼並行，共用階段 A 的人物刪除／還原歷史；不改 Gallery
embeddings 或模型相容性契約。詳見[來源移轉](SOURCE_TRANSFER_zh.md)。

### 特徵管理

Gallery Manager 以目前 model bundle、Frame 長度和點雲處理版本顯示資料。可停用或恢復單一來源／pass，也可停用某人物在目前相容性範圍內的全部特徵。

## 即時效率測試

效能頁沿用正式的 Azure Kinect、人物偵測、人體點雲前處理、模型與 Gallery 流程，但不建立影像預覽。pipeline thread 在 UI queue 之前把每幀的 scalar telemetry 送入獨立 recorder，因此 Tk 更新較慢或 queue 捨棄舊 snapshot 時不會漏掉統計幀。

模型 encode 與 Gallery match 分開計時；正式測量在第一個有效人體點雲後先暖機，再依指定秒數收集分布。recorder 只保留數值 samples、狀態計數與執行設定，不持有或序列化 RGB、Depth、點雲及 embedding。靜坐報告標記為 `stationary_efficiency_only`，禁止作為辨識準確率證據。

## 驗證狀態

- Unknown-rejection threshold calibration: **TBD**
- Azure Kinect hardware FPS: **TBD**
- Long-duration stability: **TBD**
- Person-mask alignment: **TBD**
