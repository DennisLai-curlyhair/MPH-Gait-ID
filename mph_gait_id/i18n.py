from __future__ import annotations

import json
import locale
import os
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import ttk


SUPPORTED_LOCALES = ("zh_TW", "en")
LANGUAGE_CHOICES = {
    "繁體中文": "zh_TW",
    "English": "en",
}


MESSAGES: dict[str, dict[str, str]] = {
    "nav.sources": {"zh_TW": "註冊來源點雲", "en": "Enrollment sources"},
    "sources.opt_in": {
        "zh_TW": "同意保存註冊前景點雲",
        "en": "Consent to save enrollment foreground points",
    },
    "sources.committing": {
        "zh_TW": "正在完成來源與 Gallery 操作，請稍候。",
        "en": "Completing source and Gallery operation; please wait.",
    },
    "app.title": {
        "zh_TW": "MPH-Gait ID 點雲步態身分辨識系統",
        "en": "MPH-Gait ID",
    },
    "language": {"zh_TW": "語言", "en": "Language"},
    "nav.offline": {"zh_TW": "離線註冊／辨識", "en": "Offline registration / recognition"},
    "nav.realtime": {"zh_TW": "Azure Kinect 即時模式", "en": "Real-time Azure Kinect"},
    "nav.gallery": {"zh_TW": "Gallery 管理", "en": "Gallery Manager"},
    "nav.benchmark": {"zh_TW": "即時效率測試", "en": "Live Performance Benchmark"},
    "model.management": {"zh_TW": "模型管理", "en": "Model management"},
    "model.input_type": {"zh_TW": "輸入類型", "en": "Input type"},
    "model.method": {"zh_TW": "模型方法", "en": "Model method"},
    "model.bundle": {"zh_TW": "權重 / Bundle", "en": "Checkpoint / Bundle"},
    "model.import": {"zh_TW": "匯入權重", "en": "Import checkpoint"},
    "rescan": {"zh_TW": "重新掃描", "en": "Rescan"},
    "refresh": {"zh_TW": "重新整理", "en": "Refresh"},
    "browse": {"zh_TW": "瀏覽", "en": "Browse"},
    "reset": {"zh_TW": "重設", "en": "Reset"},
    "stop": {"zh_TW": "停止", "en": "Stop"},
    "source.group": {"zh_TW": "資料來源", "en": "Data source"},
    "source.input": {"zh_TW": "輸入來源", "en": "Input source"},
    "source.add_folder": {"zh_TW": "加入序列／人物資料夾", "en": "Add sequence / person folder"},
    "source.remove": {"zh_TW": "移除選取", "en": "Remove selected"},
    "source.clear": {"zh_TW": "清空", "en": "Clear"},
    "open_set": {"zh_TW": "Open-set 設定", "en": "Open-set settings"},
    "unknown_threshold": {"zh_TW": "Unknown 門檻", "en": "Unknown threshold"},
    "min_margin": {"zh_TW": "第一／二名最小差距", "en": "Minimum top-2 margin"},
    "threshold.provisional": {
        "zh_TW": "目前為依模型與 T 分開的暫定門檻；正式部署前需用未知人物資料校正。",
        "en": "Thresholds are provisional per model and T; calibrate with unknown identities before deployment.",
    },
    "offline.enroll": {"zh_TW": "人物註冊", "en": "Identity enrollment"},
    "offline.recognize": {"zh_TW": "身分辨識", "en": "Identity recognition"},
    "offline.start_enroll": {"zh_TW": "開始註冊", "en": "Start enrollment"},
    "offline.start_recognize": {"zh_TW": "開始辨識", "en": "Start recognition"},
    "gallery.database": {"zh_TW": "Gallery 資料庫", "en": "Gallery database"},
    "gallery.registered": {"zh_TW": "已註冊人物", "en": "Registered identities"},
    "gallery.sessions": {"zh_TW": "註冊 sessions 與 passes", "en": "Registration sessions and passes"},
    "gallery.scope": {"zh_TW": "Gallery 範圍", "en": "Gallery scope"},
    "gallery.model_bundle": {"zh_TW": "模型 Bundle", "en": "Model bundle"},
    "gallery.frame_length": {"zh_TW": "Frame 長度 T", "en": "Frame length T"},
    "gallery.processing_version": {"zh_TW": "點雲處理版本", "en": "Point-cloud processing version"},
    "gallery.deactivate_pass": {"zh_TW": "停用選取 pass", "en": "Deactivate selected pass"},
    "gallery.reactivate_pass": {"zh_TW": "恢復選取 pass", "en": "Reactivate selected pass"},
    "gallery.rename_person": {"zh_TW": "修改人物名稱（所有模型）", "en": "Rename person (all models)"},
    "gallery.delete_person": {"zh_TW": "刪除人物及全部模型特徵", "en": "Delete person and ALL model embeddings"},
    "gallery.delete_fragment": {"zh_TW": "永久刪除選取片段", "en": "Permanently delete selected fragment"},
    "gallery.all_people": {"zh_TW": "顯示所有人物（跨模型，含無特徵者）", "en": "All people (all models, including empty entries)"},
    "gallery.delete_scope_hint": {"zh_TW": "刪除範圍：選取列的全部特徵", "en": "Deletion scope: all embeddings in the selected row"},
    "gallery.finish_before_edit": {
        "zh_TW": "請先完成或放棄待檢查的註冊片段，再修改或匯出／匯入 Gallery。",
        "en": "Commit or abandon the pending enrollment review before editing or transferring Gallery.",
    },
    "realtime.operation": {"zh_TW": "即時操作", "en": "Realtime operation"},
    "realtime.recognition": {"zh_TW": "即時辨識", "en": "Realtime recognition"},
    "realtime.enrollment": {"zh_TW": "即時註冊", "en": "Realtime enrollment"},
    "source.azure": {"zh_TW": "Azure Kinect DK", "en": "Azure Kinect DK"},
    "source.replay": {"zh_TW": "Replay／錄製 RGB-D", "en": "Replay / recorded RGB-D"},
    "detector.yolo": {"zh_TW": "YOLO 人物 bbox（快速）", "en": "YOLO person bbox (fast)"},
    "detector.yolo_seg": {"zh_TW": "YOLO 人物分割（快速 mask）", "en": "YOLO person segmentation (fast mask)"},
    "detector.yolo_sam": {"zh_TW": "YOLO + SAM 人物 mask（精細）", "en": "YOLO + SAM person mask (precise)"},
    "detector.replay": {"zh_TW": "Replay HOG + depth fallback（診斷）", "en": "Replay HOG + depth fallback (diagnostic)"},
    "direction.left_right": {"zh_TW": "左 → 右", "en": "Left → Right"},
    "direction.right_left": {"zh_TW": "右 → 左", "en": "Right → Left"},
    "direction.toward": {"zh_TW": "朝向相機", "en": "Toward camera"},
    "direction.away": {"zh_TW": "遠離相機", "en": "Away from camera"},
    "direction.unspecified": {"zh_TW": "未指定", "en": "Unspecified"},
    "direction.front_facing": {
        "zh_TW": "正面／面向相機（模型支援）",
        "en": "Front-facing / facing camera (model-supported)",
    },
    "review.include": {"zh_TW": "納入註冊", "en": "Include"},
    "recognition": {"zh_TW": "辨識", "en": "Recognition"},
    "enrollment": {"zh_TW": "註冊", "en": "Enrollment"},
    "realtime.source": {"zh_TW": "即時資料來源", "en": "Realtime source"},
    "realtime.detector": {"zh_TW": "人物偵測器", "en": "Person detector"},
    "realtime.check_device": {"zh_TW": "檢查 Azure Kinect", "en": "Check Azure Kinect"},
    "realtime.local_weights": {"zh_TW": "本機權重", "en": "Local weights"},
    "realtime.sam_weights": {"zh_TW": "SAM 權重", "en": "SAM weights"},
    "realtime.model": {"zh_TW": "模型與 checkpoint", "en": "Model and checkpoint"},
    "realtime.settings": {"zh_TW": "推論設定", "en": "Inference settings"},
    "realtime.start": {"zh_TW": "開始即時辨識", "en": "Start realtime recognition"},
    "realtime.start_enroll": {"zh_TW": "開始引導式註冊 session", "en": "Start guided enrollment session"},
    "realtime.start_pass": {"zh_TW": "開始 pass", "en": "Start pass"},
    "realtime.end_pass": {"zh_TW": "結束 pass", "en": "End pass"},
    "realtime.discard_pass": {"zh_TW": "捨棄 pass", "en": "Discard pass"},
    "realtime.finish_review": {"zh_TW": "完成並檢查", "en": "Finish & review"},
    "realtime.loop": {"zh_TW": "循環播放 replay", "en": "Loop replay"},
    "realtime.rgb": {"zh_TW": "RGB 人物偵測", "en": "RGB detection"},
    "realtime.cloud": {"zh_TW": "濾除背景後的人體點雲", "en": "Filtered person point cloud"},
    "realtime.rgb_pointcloud": {
        "zh_TW": "在 RGB 顯示人體點雲（僅預覽）",
        "en": "Show person point cloud on RGB (display only)",
    },
    "realtime.allow_multi_enrollment": {
        "zh_TW": "多人時仍允許註冊（只使用主要人物）",
        "en": "Allow enrollment with multiple people (primary person only)",
    },
    "telemetry.pipeline_fps": {"zh_TW": "Pipeline FPS", "en": "Pipeline FPS"},
    "telemetry.read_fps": {"zh_TW": "Frame 讀取 FPS", "en": "Frame-read FPS"},
    "telemetry.sampling_fps": {"zh_TW": "有效取樣 FPS", "en": "Valid sampling FPS"},
    "telemetry.window_span": {"zh_TW": "Window 時長", "en": "Window span"},
    "telemetry.gap": {"zh_TW": "平均／最大間隔", "en": "Mean / max gap"},
    "telemetry.buffer": {"zh_TW": "Frame buffer", "en": "Frame buffer"},
    "telemetry.points": {"zh_TW": "人體點數", "en": "Person points"},
    "telemetry.detection": {"zh_TW": "偵測 / SAM", "en": "Detection / SAM"},
    "telemetry.inference": {"zh_TW": "步態推論", "en": "Gait inference"},
    "state.ready": {"zh_TW": "準備就緒", "en": "Ready"},
    "state.no_result": {"zh_TW": "尚無即時辨識結果", "en": "No realtime result"},
    "state.wait_source": {"zh_TW": "等待資料來源", "en": "Waiting for source"},
    "state.gallery_empty": {"zh_TW": "Gallery 尚未載入", "en": "Gallery not loaded"},
    "candidate.rank": {"zh_TW": "排名", "en": "Rank"},
    "candidate.name": {"zh_TW": "姓名", "en": "Name"},
    "candidate.score": {"zh_TW": "相似度", "en": "Similarity"},
    "candidate.clips": {"zh_TW": "Gallery clips", "en": "Gallery clips"},
    "close": {"zh_TW": "關閉", "en": "Close"},
    "offline.title": {
        "zh_TW": "MPH-Gait ID 點雲步態身分辨識系統",
        "en": "MPH-Gait ID",
    },
    "offline.ready": {"zh_TW": "系統就緒", "en": "System ready"},
    "offline.no_source": {"zh_TW": "尚未選擇來源", "en": "No source selected"},
    "offline.no_person": {"zh_TW": "尚未從來源偵測人物", "en": "No identity detected from source"},
    "offline.no_result": {"zh_TW": "尚無辨識結果", "en": "No recognition result"},
    "offline.clip_features": {"zh_TW": "Clip 特徵", "en": "Clip embeddings"},
    "offline.source_features": {"zh_TW": "來源數 / Clip 特徵數", "en": "Sources / clip embeddings"},
    "column.name": {"zh_TW": "名稱", "en": "Name"},
    "column.time": {"zh_TW": "時間", "en": "Time"},
    "column.operation": {"zh_TW": "操作", "en": "Operation"},
    "column.identity": {"zh_TW": "人物", "en": "Identity"},
    "column.source": {"zh_TW": "來源", "en": "Source"},
    "column.window": {"zh_TW": "視窗", "en": "Window"},
    "column.frame_range": {"zh_TW": "Frame 範圍", "en": "Frame range"},
    "column.candidate": {"zh_TW": "候選人", "en": "Candidate"},
    "benchmark.privacy": {
        "zh_TW": "只保存數值統計，不保存 RGB、深度、點雲或特徵",
        "en": "Numeric telemetry only; RGB, depth, point clouds, and embeddings are not saved",
    },
    "benchmark.settings": {"zh_TW": "測試設定", "en": "Benchmark settings"},
    "benchmark.model": {"zh_TW": "點雲模型", "en": "Point-cloud model"},
    "benchmark.detector": {"zh_TW": "人體偵測", "en": "Person detector"},
    "benchmark.clip_length": {"zh_TW": "序列長度 T", "en": "Clip length T"},
    "benchmark.warmup": {"zh_TW": "暖機秒數", "en": "Warm-up seconds"},
    "benchmark.duration": {"zh_TW": "測量秒數", "en": "Measure seconds"},
    "benchmark.subject_state": {"zh_TW": "受測狀態", "en": "Subject state"},
    "benchmark.stationary": {"zh_TW": "靜坐（只測效率）", "en": "Stationary (efficiency only)"},
    "benchmark.walking": {"zh_TW": "行走（觀察結果行為）", "en": "Walking (observe result behavior)"},
    "benchmark.expected_id": {"zh_TW": "預期 Person ID（選填）", "en": "Expected Person ID (optional)"},
    "benchmark.start": {"zh_TW": "開始效率測試", "en": "Start benchmark"},
    "benchmark.stop": {"zh_TW": "停止並保留部分結果", "en": "Stop and keep partial result"},
    "benchmark.export": {"zh_TW": "匯出選取報告", "en": "Export selected report"},
    "benchmark.live": {"zh_TW": "即時監看（不顯示影像）", "en": "Live telemetry (no images)"},
    "benchmark.results": {"zh_TW": "本次程式的測試結果", "en": "Benchmark results in this app session"},
    "benchmark.empty_detail": {
        "zh_TW": "完成測試後，這裡會顯示報告摘要與自動保存位置。",
        "en": "A report summary and automatic save path appear here after completion.",
    },
    "benchmark.ready": {
        "zh_TW": "準備就緒。不會保存 RGB、深度或點雲影像。",
        "en": "Ready. No RGB, depth, or point-cloud frames will be saved.",
    },
    "benchmark.idle": {"zh_TW": "待機", "en": "Idle"},
    "benchmark.gallery_empty": {"zh_TW": "目前模型 Gallery：-", "en": "Current model Gallery: -"},
    "benchmark.opening": {"zh_TW": "正在開啟 Kinect 與載入模型", "en": "Opening Kinect and loading model"},
    "benchmark.warmup_hint": {
        "zh_TW": "偵測到有效人體點雲後才會開始暖機計時",
        "en": "Warm-up starts after the first valid person point cloud",
    },
    "benchmark.waiting": {"zh_TW": "等待有效人體點雲", "en": "Waiting for a valid person point cloud"},
    "benchmark.warming": {"zh_TW": "暖機中", "en": "Warm-up"},
    "benchmark.complete": {"zh_TW": "測量完成", "en": "Benchmark complete"},
    "benchmark.measuring": {"zh_TW": "正式測量中", "en": "Measuring"},
    "benchmark.stopped": {"zh_TW": "已停止，沒有可保存的正式測量資料", "en": "Stopped; no measured samples to save"},
    "benchmark.no_report": {
        "zh_TW": "測量開始前即停止，因此沒有建立報告",
        "en": "Stopped before measurement began; no report was created",
    },
    "benchmark.partial": {"zh_TW": "部分結果已保存", "en": "Partial result saved"},
    "benchmark.metric.pipeline": {"zh_TW": "端到端", "en": "End-to-end"},
    "benchmark.metric.valid": {"zh_TW": "有效點雲", "en": "Valid point cloud"},
    "benchmark.metric.read": {"zh_TW": "Kinect 讀取", "en": "Kinect read"},
    "benchmark.metric.detect": {"zh_TW": "人物偵測", "en": "Detection"},
    "benchmark.metric.preprocess": {"zh_TW": "點雲前處理", "en": "Point preprocess"},
    "benchmark.metric.model": {"zh_TW": "模型編碼", "en": "Model encode"},
    "benchmark.metric.gallery": {"zh_TW": "Gallery 比對", "en": "Gallery match"},
    "benchmark.metric.points": {"zh_TW": "人體點數", "en": "Person points"},
    "benchmark.column.model": {"zh_TW": "模型", "en": "Model"},
    "benchmark.column.detector": {"zh_TW": "偵測器", "en": "Detector"},
    "benchmark.column.mode": {"zh_TW": "模式", "en": "Mode"},
    "benchmark.column.seconds": {"zh_TW": "秒數", "en": "Seconds"},
    "benchmark.column.fps": {"zh_TW": "端到端 FPS", "en": "E2E FPS"},
    "benchmark.column.valid_fps": {"zh_TW": "有效 FPS", "en": "Valid FPS"},
    "benchmark.column.model_p50": {"zh_TW": "模型 P50", "en": "Model P50"},
    "benchmark.column.model_p95": {"zh_TW": "模型 P95", "en": "Model P95"},
    "benchmark.column.gallery_p50": {"zh_TW": "Gallery P50", "en": "Gallery P50"},
    "benchmark.column.valid_pct": {"zh_TW": "有效 %", "en": "Valid %"},
    "device": {"zh_TW": "運算裝置", "en": "Device"},
    "source.multi_hint": {
        "zh_TW": "註冊可加入多筆；辨識時請選取其中一筆",
        "en": "Enrollment accepts multiple sources; select one source for recognition",
    },
    "threshold.sequence_rule": {
        "zh_TW": "判定依據：整部來源所有 windows 的平均身分分數；單一段過線不會直接通過。",
        "en": "Decision rule: average identity score over all sequence windows; one passing window is not enough.",
    },
    "person.id": {"zh_TW": "Person ID", "en": "Person ID"},
    "person.name": {"zh_TW": "人物顯示名稱", "en": "Display name"},
    "person.note": {
        "zh_TW": "人物備註（服裝與拍攝條件由來源自動記錄）",
        "en": "Identity note (clothing and sequence conditions are recorded automatically)",
    },
    "source.autofill": {"zh_TW": "從來源自動填入", "en": "Autofill from source"},
    "candidate.people": {"zh_TW": "候選人物", "en": "Identity candidates"},
    "window.results": {"zh_TW": "逐視窗結果", "en": "Per-window results"},
    "session.history": {"zh_TW": "工作階段歷史", "en": "Session history"},
    "run.log": {"zh_TW": "執行紀錄", "en": "Run log"},
    "result.load": {"zh_TW": "載入選取結果", "en": "Load selected result"},
    "gallery.manage_person": {"zh_TW": "管理選取人物", "en": "Manage selected identity"},
    "gallery.source": {"zh_TW": "註冊來源", "en": "Enrollment source"},
    "gallery.fingerprint": {"zh_TW": "內容指紋", "en": "Content fingerprint"},
    "gallery.embedding_count": {"zh_TW": "Clip 特徵數", "en": "Clip embeddings"},
    "gallery.disable_source": {"zh_TW": "停用選取來源", "en": "Deactivate selected source"},
    "gallery.clear_model": {
        "zh_TW": "停用該人物於目前權重的全部特徵",
        "en": "Deactivate this identity's embeddings for the current checkpoint",
    },
    "gallery.description": {
        "zh_TW": "人物、註冊片段與資料移轉",
        "en": "Identities, enrollment fragments and data transfer",
    },
    "gallery.legacy": {
        "zh_TW": "舊資料可依來源列永久刪除；啟停 pass 需要 session／pass ID。",
        "en": "Legacy rows can be deleted by source; pass activation requires session/pass IDs.",
    },
    "realtime.session_timing": {
        "zh_TW": "最長 session／pass 暖身（秒）",
        "en": "Max session / pass warm-up (s)",
    },
    "realtime.embedding_counts": {
        "zh_TW": "最少／保存特徵數",
        "en": "Min. / stored embeddings",
    },
    "realtime.autofill": {
        "zh_TW": "由 replay 名稱自動填入",
        "en": "Autofill from replay name",
    },
    "realtime.guided_passes": {"zh_TW": "引導式單程 passes", "en": "Guided one-way passes"},
    "realtime.sam_interval": {"zh_TW": "SAM 更新間隔（frames）", "en": "SAM refresh interval (frames)"},
    "realtime.open_offline": {"zh_TW": "開啟離線批次頁", "en": "Open Offline batch"},
    "realtime.manage_gallery": {"zh_TW": "管理 Gallery", "en": "Manage Gallery"},
    "realtime.frame_length": {"zh_TW": "Frame 長度", "en": "Frame length"},
    "realtime.stride": {"zh_TW": "推論 stride", "en": "Inference stride"},
    "realtime.replay_fps": {"zh_TW": "Replay FPS", "en": "Replay FPS"},
    "realtime.yolo_conf": {"zh_TW": "YOLO 信心門檻", "en": "YOLO confidence"},
    "realtime.min_margin": {"zh_TW": "最小 top-2 差距", "en": "Min. top-2 margin"},
    "realtime.display_name": {"zh_TW": "顯示名稱", "en": "Display name"},
    "realtime.note": {"zh_TW": "備註", "en": "Note"},
    "pass": {"zh_TW": "Pass", "en": "Pass"},
    "direction": {"zh_TW": "方向", "en": "Direction"},
    "frames": {"zh_TW": "Frames", "en": "Frames"},
    "state": {"zh_TW": "狀態", "en": "State"},
    "realtime.fixed_view": {"zh_TW": "固定 2.4 m 視野", "en": "Fixed 2.4 m view"},
    "realtime.foreground_cloud": {"zh_TW": "人體前景點雲", "en": "Foreground point cloud"},
    "realtime.open_review": {"zh_TW": "開啟檢查", "en": "Open review"},
    "realtime.no_pass": {"zh_TW": "尚未擷取任何 pass。", "en": "No pass was captured."},
    "realtime.abandon": {"zh_TW": "放棄 session", "en": "Abandon session"},
    "realtime.close_reopen": {"zh_TW": "關閉（可在本程式重開）", "en": "Close (reopen in this app)"},
    "realtime.continue_passes": {
        "zh_TW": "返回擷取並繼續加入 pass",
        "en": "Continue adding passes",
    },
    "realtime.commit": {"zh_TW": "提交選取 passes", "en": "Commit selected passes"},
    "player.empty": {
        "zh_TW": "選擇點雲資料夾後開始註冊／辨識",
        "en": "Select a point-cloud folder, then start enrollment or recognition",
    },
    "player.play": {"zh_TW": "播放", "en": "Play"},
    "player.pause": {"zh_TW": "暫停", "en": "Pause"},
    "player.previous": {"zh_TW": "上一格", "en": "Previous"},
    "player.next": {"zh_TW": "下一格", "en": "Next"},
    "player.view": {"zh_TW": "預覽視角", "en": "Preview orientation"},
    "player.rotate_left": {"zh_TW": "左轉", "en": "Rotate left"},
    "player.rotate_right": {"zh_TW": "右轉", "en": "Rotate right"},
    "player.flip_h": {"zh_TW": "水平鏡像", "en": "Mirror horizontally"},
    "player.flip_v": {"zh_TW": "垂直翻轉", "en": "Flip vertically"},
    "player.position": {"zh_TW": "畫面位置", "en": "View position"},
    "player.left": {"zh_TW": "左", "en": "Left"},
    "player.right": {"zh_TW": "右", "en": "Right"},
    "player.up": {"zh_TW": "上", "en": "Up"},
    "player.down": {"zh_TW": "下", "en": "Down"},
    "player.zoom_in": {"zh_TW": "放大", "en": "Zoom in"},
    "player.zoom_out": {"zh_TW": "縮小", "en": "Zoom out"},
    "player.overlay": {"zh_TW": "顯示畫面標註", "en": "Show result overlay"},
    "player.overlay_position": {"zh_TW": "標註位置", "en": "Overlay position"},
    "player.cloud_view": {"zh_TW": "點雲視角", "en": "Point-cloud view"},
    "player.no_visual": {"zh_TW": "尚無可視化結果", "en": "No visualization result"},
}


def _default_settings_path() -> Path:
    override = os.environ.get("MPH_GAIT_ID_SETTINGS", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        root = Path(os.environ.get("APPDATA", Path.home()))
    elif sys_platform() == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "mph_gait_id" / "settings.json"


def sys_platform() -> str:
    import sys

    return sys.platform


def _system_locale() -> str:
    language = (locale.getlocale()[0] or "").lower()
    return "zh_TW" if language.startswith("zh") else "en"


class I18n:
    def __init__(self, configured_locale: str = "auto") -> None:
        self.settings_path = _default_settings_path()
        saved = self._load_settings().get("language")
        requested = str(saved or configured_locale or "auto")
        self.locale = (
            _system_locale()
            if requested == "auto"
            else requested if requested in SUPPORTED_LOCALES else "en"
        )
        self._aliases: dict[str, str] = {}
        for key, translations in MESSAGES.items():
            self._aliases[key] = key
            for value in translations.values():
                self._aliases[value] = key

    def _load_settings(self) -> dict[str, Any]:
        try:
            value = json.loads(self.settings_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (FileNotFoundError, OSError, ValueError):
            return {}

    def set_locale(self, value: str) -> None:
        if value not in SUPPORTED_LOCALES:
            raise ValueError(f"Unsupported language: {value}")
        self.locale = value
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            self.settings_path.write_text(
                json.dumps({"language": value}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass

    def tr(self, key_or_text: str, **values: Any) -> str:
        source = str(key_or_text)
        key = self._aliases.get(source, source)
        translated = MESSAGES.get(key, {}).get(self.locale, source)
        try:
            return translated.format(**values)
        except (KeyError, ValueError):
            return translated

    def language_choice(self) -> str:
        return next(
            label for label, value in LANGUAGE_CHOICES.items() if value == self.locale
        )

    def apply(self, root: tk.Misc) -> None:
        self._apply_widget(root)
        for child in root.winfo_children():
            self.apply(child)

    def _apply_widget(self, widget: tk.Misc) -> None:
        try:
            current = str(widget.cget("text"))
            translated = self.tr(current)
            if translated != current:
                widget.configure(text=translated)
        except (tk.TclError, AttributeError):
            pass
        if isinstance(widget, ttk.Label):
            try:
                variable_name = str(widget.cget("textvariable"))
                if variable_name:
                    current = str(widget.getvar(variable_name))
                    translated = self.tr(current)
                    if translated != current:
                        widget.setvar(variable_name, translated)
            except tk.TclError:
                pass
        if isinstance(widget, ttk.Notebook):
            for tab_id in widget.tabs():
                current = str(widget.tab(tab_id, "text"))
                widget.tab(tab_id, text=self.tr(current))
        if isinstance(widget, ttk.Treeview):
            columns = ["#0", *list(widget.cget("columns"))]
            for column in columns:
                try:
                    current = str(widget.heading(column, "text"))
                    widget.heading(column, text=self.tr(current))
                except tk.TclError:
                    pass
