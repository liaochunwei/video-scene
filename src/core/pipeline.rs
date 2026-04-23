//! 视频处理管线模块
//!
//! 定义从原始视频到结构化索引数据的完整处理流程：
//! 1. 视频分析（获取时长/分辨率/帧率等元信息）
//! 2. 场景检测（本地 PySceneDetect 或云端 VLM API）
//! 3. 帧提取（按场景边界抽帧）
//! 4. AI 模型批量推理（人脸检测、物体检测、VLM 描述、文本/图像编码）
//! 5. 数据入库（视频、片段、检测记录写入 SQLite）
//! 6. 保存向量索引（人脸/场景/图像 HNSW 索引落盘）
//!
//! 提供两种模式：
//! - `process_video`：本地管线，场景切分用 PySceneDetect，描述用本地 VLM
//! - `process_video_vlm_api`：云端管线，场景切分和描述由 VLM API 一步完成

use std::path::Path;

use crate::config::Settings;
use crate::error::{Result, VideoSceneError};
use crate::models::{Detection, Segment, Video};
use crate::plugins::image_text_understanding::DescriptionCategory;
use crate::plugins::ProgressMessage;
use crate::preprocess::{
    scene_detector::{PySceneDetector, SceneBoundary, SceneDetector, SingleSceneDetector},
    video_analyzer, extract_frames,
};
use crate::storage::{Database, FileStore, SceneIndices, VectorIndex};

/// 单个视频的管线处理结果，包含视频元信息、场景片段和检测记录。
pub struct PipelineResult {
    pub video: Video,
    pub segments: Vec<Segment>,
    pub detections: Vec<Detection>,
}

/// 本地管线模式：分析视频 → 检测场景 → 抽帧 → 批量推理 → 入库。
#[allow(clippy::too_many_arguments)]
pub fn process_video(
    video_path: &Path,
    settings: &Settings,
    db: &Database,
    file_store: &FileStore,
    face_index: &mut VectorIndex,
    scene_indices: &mut SceneIndices,
    image_index: &mut VectorIndex,
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<PipelineResult> {
    let no_progress: &dyn Fn(ProgressMessage) = &|_| {};

    // ---- 步骤 1：分析视频元信息 ----
    tracing::info!("[1/6] Analyzing video: {}", video_path.display());
    let info = video_analyzer::analyze_video(video_path)?;
    tracing::info!("[1/6] Video: {}x{} {:.1}s {:.0}fps", info.width, info.height, info.duration, info.fps);

    let video = Video::new(
        video_path.to_string_lossy().to_string(),
        video_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
        info.duration,
        info.width,
        info.height,
    );

    // ---- 步骤 2：检测场景边界 ----
    tracing::info!("[2/6] Detecting scenes...");
    // 根据配置选择场景检测器；检测失败时退化为整段视频作为单一场景
    let boundaries: Vec<SceneBoundary> = if settings.video.scene_detection.detector == "single" {
        SingleSceneDetector.detect(video_path, no_progress)?
    } else {
        let detector = PySceneDetector::new(
            settings.video.scene_detection.detector.clone(),
            settings.video.scene_detection.threshold,
        );
        match detector.detect(video_path, progress_cb) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!("[2/6] Scene detection failed ({}), using single scene fallback", e);
                SingleSceneDetector.detect(video_path, no_progress)?
            }
        }
    };
    tracing::info!("[2/6] Found {} scenes", boundaries.len());

    // 基于场景数量和历史耗时估算剩余处理时间
    let est = crate::core::timing::estimate_remaining(file_store.base_dir(), boundaries.len());
    if !est.is_empty() {
        tracing::info!("[2/6] Estimated remaining: {}", est);
    }

    // ---- 步骤 3：为每个场景提取帧图像 ----
    tracing::info!("[3/6] Extracting frames...");
    let mut segments = Vec::new();
    let mut all_frames = Vec::new(); // (片段索引, 帧信息)
    let temp_dir = std::env::temp_dir().join("video-scene-frames");

    for (i, boundary) in boundaries.iter().enumerate() {
        let segment = Segment::new(
            video.id,
            boundary.start,
            boundary.end,
            String::new(),
        );

        // 根据场景时长自适应决定抽帧密度
        let timestamps = calculate_frame_timestamps(boundary);

        let frames = extract_frames(
            video_path,
            &timestamps,
            &temp_dir.join(video.id.to_string()).join(format!("seg_{:03}", i)),
            settings.video.preprocessing.target_short_edge,
            settings.video.preprocessing.frame_quality,
        )?;

        for frame in &frames {
            all_frames.push((i, frame.clone()));
        }

        segments.push(segment);
    }
    tracing::info!("[3/6] Extracted {} frames from {} segments", all_frames.len(), segments.len());

    // ---- 步骤 4：批量 AI 推理 ----
    let total_frames = all_frames.len();
    let frame_paths: Vec<String> = all_frames.iter()
        .map(|(_, f)| f.path.to_string_lossy().to_string())
        .collect();

    // 4a. 保存关键帧到 FileStore（纯文件 I/O，无需模型推理）
    // 每个片段保存第一帧作为缩略图/关键帧
    tracing::info!("[4/6] Saving keyframes...");
    for (seg_idx, frame) in &all_frames {
        let segment = &mut segments[*seg_idx];
        if segment.keyframe_path.is_empty() {
            let keyframe_data = std::fs::read(&frame.path)
                .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
            let keyframe_path = file_store.save_keyframe(
                &video.id.to_string(),
                &segment.id.to_string(),
                &keyframe_data,
            )?;
            segment.keyframe_path = keyframe_path.to_string_lossy().to_string();
        }
    }

    // 4b. 人脸检测（批量）
    // 优先使用批量接口减少模型加载开销；失败时退化为逐帧检测
    progress_cb(ProgressMessage {
        id: String::new(),
        message: "[4/6] Face detection...".to_string(),
        current: 0,
        total: total_frames,
    });
    let mut detections = Vec::new();
    let face_results: Vec<(usize, Vec<crate::plugins::face::FaceDetection>)> = match
        crate::plugins::face::detect_faces_batch(&frame_paths, settings.index.face.min_confidence, progress_cb)
    {
        Ok(batch) => batch.into_iter()
            .filter_map(|(path, faces)| {
                // 批量接口返回的是 (路径, 检测结果)，需映射回帧索引
                frame_paths.iter().position(|p| *p == path).map(|idx| (idx, faces))
            })
            .collect(),
        Err(e) => {
            // 批量失败时退化为逐帧调用，牺牲速度但保证不中断
            tracing::warn!("Batch face detection failed: {}, falling back to per-frame", e);
            let mut results = Vec::new();
            for (i, path) in frame_paths.iter().enumerate() {
                progress_cb(ProgressMessage {
                    id: String::new(),
                    message: "[4/6] Face detection (fallback)...".to_string(),
                    current: i + 1,
                    total: total_frames,
                });
                match crate::plugins::face::detect_faces(path, settings.index.face.min_confidence, no_progress) {
                    Ok(faces) => results.push((i, faces)),
                    Err(e2) => tracing::warn!("Face detection failed for frame {}: {}", path, e2),
                }
            }
            results
        }
    };
    // 将人脸检测结果转为 Detection 记录，同时写入人脸向量索引
    for (frame_idx, faces) in &face_results {
        let seg_idx = all_frames[*frame_idx].0;
        let segment = &segments[seg_idx];
        for face in faces {
            // 质量低于阈值的脸（模糊、侧脸等）跳过，避免污染人脸索引
            if face.quality >= settings.index.face.min_quality as f32 {
                let detection = Detection::new_face(
                    segment.id,
                    String::new(),
                    face.confidence,
                    Some(face.bbox.clone()),
                    face.feature.clone(),
                );
                // 同时加入 HNSW 向量索引，用于后续人脸搜索
                face_index.add(detection.id.to_string(), face.feature.clone());
                detections.push(detection);
            }
        }
    }
    tracing::info!("[4/6] Face detection: {} faces in {} frames", detections.len(), face_results.len());

    // 4c. 物体检测（批量）
    progress_cb(ProgressMessage {
        id: String::new(),
        message: "[4/6] Object detection...".to_string(),
        current: 0,
        total: total_frames,
    });
    let obj_results: Vec<(usize, Vec<crate::plugins::object::ObjectDetection>)> = match
        crate::plugins::object::detect_objects_batch(&frame_paths, settings.index.object.min_confidence, None, progress_cb)
    {
        Ok(batch) => batch.into_iter()
            .filter_map(|(path, objects)| {
                frame_paths.iter().position(|p| *p == path).map(|idx| (idx, objects))
            })
            .collect(),
        Err(e) => {
            tracing::warn!("Batch object detection failed: {}, falling back to per-frame", e);
            let mut results = Vec::new();
            for (i, path) in frame_paths.iter().enumerate() {
                progress_cb(ProgressMessage {
                    id: String::new(),
                    message: "[4/6] Object detection (fallback)...".to_string(),
                    current: i + 1,
                    total: total_frames,
                });
                match crate::plugins::object::detect_objects(path, settings.index.object.min_confidence, None, no_progress) {
                    Ok(objects) => results.push((i, objects)),
                    Err(e2) => tracing::warn!("Object detection failed for frame {}: {}", path, e2),
                }
            }
            results
        }
    };
    let mut obj_detections = Vec::new();
    for (frame_idx, objects) in &obj_results {
        let seg_idx = all_frames[*frame_idx].0;
        let segment = &segments[seg_idx];
        for obj in objects {
            let detection = Detection::new_object(
                segment.id,
                obj.label_zh.clone(),
                obj.confidence,
                Some(obj.bbox.clone()),
            );
            obj_detections.push(detection);
        }
    }
    tracing::info!("[4/6] Object detection: {} objects in {} frames", obj_detections.len(), obj_results.len());
    detections.extend(obj_detections);

    // 4d. VLM 场景描述——仅对时长在合理范围内的片段执行
    // 过短片段信息量不足，过长片段可能已被场景检测器误切，都不适合 VLM 描述
    let min_dur = settings.index.scene.min_segment_duration;
    let max_dur = settings.index.scene.max_segment_duration;
    let vlm_eligible: Vec<bool> = segments.iter()
        .map(|s| {
            let dur = s.end_time - s.start_time;
            dur >= min_dur && dur <= max_dur
        })
        .collect();
    let vlm_count = vlm_eligible.iter().filter(|&&e| e).count();
    let skipped_count = segments.len() - vlm_count;
    if skipped_count > 0 {
        tracing::info!("[4/6] Scene encoding: skipping {} segments outside duration range [{:.1}s, {:.1}s]", skipped_count, min_dur, max_dur);
    }

    progress_cb(ProgressMessage {
        id: String::new(),
        message: "[4/6] VLM describing scenes...".to_string(),
        current: 0,
        total: vlm_count,
    });
    // 按片段分组帧路径，只为符合时长条件的片段调用 VLM
    let mut seg_frame_groups: Vec<Vec<String>> = Vec::new();
    let vlm_seg_indices: Vec<usize>; // vlm_results 索引 → segments 索引的映射
    for (seg_idx, frame) in &all_frames {
        // Ensure seg_frame_groups has enough entries
        while seg_frame_groups.len() <= *seg_idx {
            seg_frame_groups.push(Vec::new());
        }
        seg_frame_groups[*seg_idx].push(frame.path.to_string_lossy().to_string());
    }
    // 只保留符合时长条件的片段的帧组
    let vlm_groups: Vec<Vec<String>> = vlm_eligible.iter().enumerate()
        .filter(|(_, &eligible)| eligible)
        .filter_map(|(seg_idx, _)| seg_frame_groups.get(seg_idx).cloned())
        .collect();
    vlm_seg_indices = vlm_eligible.iter().enumerate()
        .filter(|(_, &eligible)| eligible)
        .map(|(seg_idx, _)| seg_idx)
        .collect();

    // VLM 批量描述，失败时用空描述填充以保证后续流程不中断
    let vlm_results_raw = if vlm_groups.iter().any(|g| !g.is_empty()) {
        match crate::plugins::image_text_understanding::describe_scenes_batch(&vlm_groups, None, progress_cb) {
            Ok(descs) => descs,
            Err(e) => {
                tracing::warn!("VLM scene description failed: {}, segments will have no description", e);
                vec![crate::plugins::image_text_understanding::StructuredDescription::default(); vlm_count]
            }
        }
    } else {
        vec![crate::plugins::image_text_understanding::StructuredDescription::default(); vlm_count]
    };
    // 将 VLM 结果映射回完整片段列表（不符合条件的片段保持空描述）
    let vlm_results: Vec<crate::plugins::image_text_understanding::StructuredDescription> = {
        let mut results = vec![crate::plugins::image_text_understanding::StructuredDescription::default(); segments.len()];
        for (vlm_idx, &seg_idx) in vlm_seg_indices.iter().enumerate() {
            results[seg_idx] = vlm_results_raw[vlm_idx].clone();
        }
        results
    };

    // 4e. Harrier 文本嵌入——将所有类别的描述文本批量编码为向量
    progress_cb(ProgressMessage {
        id: String::new(),
        message: "[4/6] Scene encoding...".to_string(),
        current: 0,
        total: segments.len(),
    });
    // 先将 VLM 描述文本写入片段记录
    for (seg_idx, desc) in vlm_results.iter().enumerate() {
        segments[seg_idx].scene_description = desc.to_full_text();
    }
    // 将所有片段的所有类别文本打平为一个列表，一次性批量编码，
    // 避免逐段调用带来的模型加载开销
    let mut all_texts: Vec<String> = Vec::new();
    let mut text_meta: Vec<(DescriptionCategory, usize)> = Vec::new(); // (类别, 片段索引)
    for category in crate::plugins::image_text_understanding::DescriptionCategory::all() {
        for (seg_idx, desc) in vlm_results.iter().enumerate() {
            let text = desc.get_text(*category);
            if !text.is_empty() {
                all_texts.push(text);
                text_meta.push((*category, seg_idx));
            }
        }
    }
    let total_encoded = if !all_texts.is_empty() {
        match crate::plugins::text_vectorization::encode_documents_batch(&all_texts, progress_cb) {
            Ok(encoded) => {
                for (i, (_text, vector)) in encoded.iter().enumerate() {
                    let (category, seg_idx) = text_meta[i];
                    let segment = &segments[seg_idx];
                    // 将向量加入对应类别的 HNSW 索引
                    scene_indices.indices.get_mut(&category).unwrap()
                        .add(segment.id.to_string(), vector.clone());
                }
                encoded.len()
            }
            Err(e) => {
                tracing::warn!("Harrier scene encoding failed: {}", e);
                0
            }
        }
    } else {
        0
    };
    tracing::info!("[4/6] Scene encoding: {} segments, {} category vectors encoded", segments.len(), total_encoded);

    // 4f. CLIP 图像编码——将所有帧图像编码为向量，用于以图搜图
    progress_cb(ProgressMessage {
        id: String::new(),
        message: "[4/6] Image encoding...".to_string(),
        current: 0,
        total: all_frames.len(),
    });
    match crate::plugins::image_text_vectorization::encode_images_batch(&frame_paths, progress_cb) {
        Ok(encoded) => {
            for (path, vector) in &encoded {
                // 找到帧对应的片段索引
                if let Some(frame_idx) = frame_paths.iter().position(|p| *p == *path) {
                    let (seg_idx, _) = all_frames[frame_idx];
                    let segment = &segments[seg_idx];
                    // 计算该帧在所属片段内的序号，构建复合 ID
                    let frame_in_seg = all_frames.iter()
                        .take(frame_idx + 1)
                        .filter(|(si, _)| *si == seg_idx)
                        .count() - 1;
                    let image_id = format!("{}_frame_{}", segment.id, frame_in_seg);
                    image_index.add(image_id, vector.clone());
                }
            }
            tracing::info!("[4/6] Image encoding: {} frames encoded", encoded.len());
        }
        Err(e) => tracing::warn!("CLIP image encoding failed: {}", e),
    }

    // ---- 步骤 5：写入数据库 ----
    tracing::info!("[5/6] Storing results in database...");
    db.insert_video(&video)?;
    for segment in &segments {
        db.insert_segment(segment)?;
    }
    for detection in &detections {
        db.insert_detection(detection)?;
    }

    // ---- 步骤 6：保存向量索引到磁盘 ----
    tracing::info!("[6/6] Saving vector indices...");
    face_index.save()?;
    scene_indices.save()?;
    image_index.save()?;

    Ok(PipelineResult {
        video,
        segments,
        detections,
    })
}

/// 根据场景时长自适应计算抽帧时间点。
///
/// 策略：
/// - < 3s：只取中间一帧（短片段信息量少，一帧足够）
/// - 3~9s：取开头、中间、结尾各一帧（捕捉场景过渡）
/// - > 9s：每隔 3s 抽一帧，并确保包含中间帧（长场景需要更多帧来覆盖内容变化）
fn calculate_frame_timestamps(boundary: &SceneBoundary) -> Vec<f32> {
    let duration = boundary.end - boundary.start;
    let mut timestamps = vec![];

    if duration < 3.0 {
        // 短场景：中间一帧即可
        timestamps.push(boundary.start + duration / 2.0);
    } else if duration < 9.0 {
        // 中等场景：开头+中间+结尾，捕捉场景起止和核心内容
        timestamps.push(boundary.start + 1.0);
        timestamps.push(boundary.start + duration / 2.0);
        timestamps.push(boundary.end - 1.0);
    } else {
        // 长场景：每 3 秒一帧，覆盖内容变化
        let mut t = boundary.start + 1.0;
        while t < boundary.end - 1.0 {
            timestamps.push(t);
            t += 3.0;
        }
        // 确保中间帧被包含，即使不在 3s 整数点上
        let mid = boundary.start + duration / 2.0;
        if !timestamps.contains(&mid) {
            timestamps.push(mid);
        }
    }

    timestamps
}

/// 云端 VLM API 模式：由云端模型同时完成场景切分和描述，跳过本地场景检测和 VLM 推理。
///
/// 流程与本地管线类似，但步骤 2 改为调用 VLM API，步骤 4d 直接使用 API 返回的描述。
#[allow(clippy::too_many_arguments)]
pub fn process_video_vlm_api(
    video_path: &Path,
    settings: &Settings,
    db: &Database,
    file_store: &FileStore,
    face_index: &mut VectorIndex,
    scene_indices: &mut SceneIndices,
    image_index: &mut VectorIndex,
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<PipelineResult> {
    let no_progress: &dyn Fn(ProgressMessage) = &|_| {};

    // 必须配置 API Key 才能使用云端模式
    if settings.plugins.vlm_api.api_key.is_empty() {
        return Err(VideoSceneError::ConfigMissing(
            "plugins.vlm_api.api_key is required for --mode video. Set it in config.toml".into()
        ));
    }

    // ---- 步骤 1：分析视频元信息 ----
    tracing::info!("[1/6] Analyzing video: {}", video_path.display());
    let info = video_analyzer::analyze_video(video_path)?;
    tracing::info!("[1/6] Video: {}x{} {:.1}s {:.0}fps", info.width, info.height, info.duration, info.fps);

    let video = Video::new(
        video_path.to_string_lossy().to_string(),
        video_path.file_name().unwrap_or_default().to_string_lossy().to_string(),
        info.duration,
        info.width,
        info.height,
    );

    // 基于视频时长估算剩余处理时间（VLM 模式下耗时与视频长度线性相关）
    let est = crate::core::timing::estimate_remaining_vlm(file_store.base_dir(), info.duration as f64);
    if !est.is_empty() {
        tracing::info!("[1/6] Estimated remaining: {}", est);
    }

    // ---- 步骤 2：调用 VLM API 进行场景切分和描述 ----
    // 云端模型直接返回带时间戳和描述的片段列表，无需本地场景检测
    tracing::info!("[2/6] Calling VLM API for segmentation...");
    progress_cb(ProgressMessage {
        id: String::new(),
        message: "[2/6] VLM API segmentation...".to_string(),
        current: 0,
        total: 1,
    });
    let vlm_segments = crate::plugins::video_understanding::describe_video(
        &video_path.to_string_lossy(),
        &settings.plugins.vlm_api.api_base,
        &settings.plugins.vlm_api.api_key,
        &settings.plugins.vlm_api.model,
        settings.plugins.vlm_api.max_pixels,
        settings.plugins.vlm_api.fps,
        progress_cb,
    )?;

    if vlm_segments.is_empty() {
        return Err(VideoSceneError::PluginExecutionError(
            "VLM API returned 0 segments".into()
        ));
    }
    tracing::info!("[2/6] VLM API returned {} segments", vlm_segments.len());

    // ---- 步骤 3：按 API 返回的片段边界提取帧 ----
    tracing::info!("[3/6] Extracting frames...");
    let mut segments = Vec::new();
    let mut all_frames = Vec::new(); // (seg_idx, ExtractedFrame)
    let temp_dir = std::env::temp_dir().join("video-scene-frames");

    for (i, vlm_seg) in vlm_segments.iter().enumerate() {
        // 将时间戳钳制在视频时长范围内，防止 API 返回越界值
        let start = vlm_seg.start_time.max(0.0).min(info.duration);
        let end = vlm_seg.end_time.max(start).min(info.duration);

        let segment = Segment::new(video.id, start, end, String::new());

        let boundary = SceneBoundary { start, end };
        let timestamps = calculate_frame_timestamps(&boundary);

        let frames = extract_frames(
            video_path,
            &timestamps,
            &temp_dir.join(video.id.to_string()).join(format!("seg_{:03}", i)),
            settings.video.preprocessing.target_short_edge,
            settings.video.preprocessing.frame_quality,
        )?;

        for frame in &frames {
            all_frames.push((i, frame.clone()));
        }

        segments.push(segment);
    }
    tracing::info!("[3/6] Extracted {} frames from {} segments", all_frames.len(), segments.len());

    // ---- 步骤 4：批量 AI 推理 ----
    let total_frames = all_frames.len();
    let frame_paths: Vec<String> = all_frames.iter()
        .map(|(_, f)| f.path.to_string_lossy().to_string())
        .collect();

    // 4a. 保存关键帧
    tracing::info!("[4/6] Saving keyframes...");
    for (seg_idx, frame) in &all_frames {
        let segment = &mut segments[*seg_idx];
        if segment.keyframe_path.is_empty() {
            let keyframe_data = std::fs::read(&frame.path)
                .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
            let keyframe_path = file_store.save_keyframe(
                &video.id.to_string(),
                &segment.id.to_string(),
                &keyframe_data,
            )?;
            segment.keyframe_path = keyframe_path.to_string_lossy().to_string();
        }
    }

    // 4b. 人脸检测（批量，失败退化为逐帧）
    progress_cb(ProgressMessage {
        id: String::new(),
        message: "[4/6] Face detection...".to_string(),
        current: 0,
        total: total_frames,
    });
    let mut detections = Vec::new();
    let face_results: Vec<(usize, Vec<crate::plugins::face::FaceDetection>)> = match
        crate::plugins::face::detect_faces_batch(&frame_paths, settings.index.face.min_confidence, progress_cb)
    {
        Ok(batch) => batch.into_iter()
            .filter_map(|(path, faces)| {
                frame_paths.iter().position(|p| *p == path).map(|idx| (idx, faces))
            })
            .collect(),
        Err(e) => {
            tracing::warn!("Batch face detection failed: {}, falling back to per-frame", e);
            let mut results = Vec::new();
            for (i, path) in frame_paths.iter().enumerate() {
                progress_cb(ProgressMessage {
                    id: String::new(),
                    message: "[4/6] Face detection (fallback)...".to_string(),
                    current: i + 1,
                    total: total_frames,
                });
                match crate::plugins::face::detect_faces(path, settings.index.face.min_confidence, no_progress) {
                    Ok(faces) => results.push((i, faces)),
                    Err(e2) => tracing::warn!("Face detection failed for frame {}: {}", path, e2),
                }
            }
            results
        }
    };
    for (frame_idx, faces) in &face_results {
        let seg_idx = all_frames[*frame_idx].0;
        let segment = &segments[seg_idx];
        for face in faces {
            if face.quality >= settings.index.face.min_quality as f32 {
                let detection = Detection::new_face(
                    segment.id,
                    String::new(),
                    face.confidence,
                    Some(face.bbox.clone()),
                    face.feature.clone(),
                );
                face_index.add(detection.id.to_string(), face.feature.clone());
                detections.push(detection);
            }
        }
    }
    tracing::info!("[4/6] Face detection: {} faces in {} frames", detections.len(), face_results.len());

    // 4c. 物体检测（批量，失败退化为逐帧）
    progress_cb(ProgressMessage {
        id: String::new(),
        message: "[4/6] Object detection...".to_string(),
        current: 0,
        total: total_frames,
    });
    let obj_results: Vec<(usize, Vec<crate::plugins::object::ObjectDetection>)> = match
        crate::plugins::object::detect_objects_batch(&frame_paths, settings.index.object.min_confidence, None, progress_cb)
    {
        Ok(batch) => batch.into_iter()
            .filter_map(|(path, objects)| {
                frame_paths.iter().position(|p| *p == path).map(|idx| (idx, objects))
            })
            .collect(),
        Err(e) => {
            tracing::warn!("Batch object detection failed: {}, falling back to per-frame", e);
            let mut results = Vec::new();
            for (i, path) in frame_paths.iter().enumerate() {
                progress_cb(ProgressMessage {
                    id: String::new(),
                    message: "[4/6] Object detection (fallback)...".to_string(),
                    current: i + 1,
                    total: total_frames,
                });
                match crate::plugins::object::detect_objects(path, settings.index.object.min_confidence, None, no_progress) {
                    Ok(objects) => results.push((i, objects)),
                    Err(e2) => tracing::warn!("Object detection failed for frame {}: {}", path, e2),
                }
            }
            results
        }
    };
    let mut obj_detections = Vec::new();
    for (frame_idx, objects) in &obj_results {
        let seg_idx = all_frames[*frame_idx].0;
        let segment = &segments[seg_idx];
        for obj in objects {
            let detection = Detection::new_object(
                segment.id,
                obj.label_zh.clone(),
                obj.confidence,
                Some(obj.bbox.clone()),
            );
            obj_detections.push(detection);
        }
    }
    tracing::info!("[4/6] Object detection: {} objects in {} frames", obj_detections.len(), obj_results.len());
    detections.extend(obj_detections);

    // 4d. VLM 描述已在步骤 2 由 API 返回，直接写入片段记录
    for (seg_idx, vlm_seg) in vlm_segments.iter().enumerate() {
        segments[seg_idx].scene_description = vlm_seg.description.to_full_text();
    }

    // 4e. Harrier 文本嵌入——与本地管线相同的批量编码逻辑
    progress_cb(ProgressMessage {
        id: String::new(),
        message: "[4/6] Scene encoding...".to_string(),
        current: 0,
        total: segments.len(),
    });
    let mut all_texts: Vec<String> = Vec::new();
    let mut text_meta: Vec<(crate::plugins::image_text_understanding::DescriptionCategory, usize)> = Vec::new();
    for category in crate::plugins::image_text_understanding::DescriptionCategory::all() {
        for (seg_idx, vlm_seg) in vlm_segments.iter().enumerate() {
            let text = vlm_seg.description.get_text(*category);
            if !text.is_empty() {
                all_texts.push(text);
                text_meta.push((*category, seg_idx));
            }
        }
    }
    let total_encoded = if !all_texts.is_empty() {
        match crate::plugins::text_vectorization::encode_documents_batch(&all_texts, progress_cb) {
            Ok(encoded) => {
                for (i, (_text, vector)) in encoded.iter().enumerate() {
                    let (category, seg_idx) = text_meta[i];
                    let segment = &segments[seg_idx];
                    scene_indices.indices.get_mut(&category).unwrap()
                        .add(segment.id.to_string(), vector.clone());
                }
                encoded.len()
            }
            Err(e) => {
                tracing::warn!("Harrier scene encoding failed: {}", e);
                0
            }
        }
    } else {
        0
    };
    tracing::info!("[4/6] Scene encoding: {} segments, {} category vectors encoded", segments.len(), total_encoded);

    // 4f. CLIP 图像编码
    progress_cb(ProgressMessage {
        id: String::new(),
        message: "[4/6] Image encoding...".to_string(),
        current: 0,
        total: all_frames.len(),
    });
    match crate::plugins::image_text_vectorization::encode_images_batch(&frame_paths, progress_cb) {
        Ok(encoded) => {
            for (path, vector) in &encoded {
                if let Some(frame_idx) = frame_paths.iter().position(|p| *p == *path) {
                    let (seg_idx, _) = all_frames[frame_idx];
                    let segment = &segments[seg_idx];
                    let frame_in_seg = all_frames.iter()
                        .take(frame_idx + 1)
                        .filter(|(si, _)| *si == seg_idx)
                        .count() - 1;
                    let image_id = format!("{}_frame_{}", segment.id, frame_in_seg);
                    image_index.add(image_id, vector.clone());
                }
            }
            tracing::info!("[4/6] Image encoding: {} frames encoded", encoded.len());
        }
        Err(e) => tracing::warn!("CLIP image encoding failed: {}", e),
    }

    // ---- 步骤 5：写入数据库 ----
    tracing::info!("[5/6] Storing results in database...");
    db.insert_video(&video)?;
    for segment in &segments {
        db.insert_segment(segment)?;
    }
    for detection in &detections {
        db.insert_detection(detection)?;
    }

    // ---- 步骤 6：保存向量索引 ----
    tracing::info!("[6/6] Saving vector indices...");
    face_index.save()?;
    scene_indices.save()?;
    image_index.save()?;

    Ok(PipelineResult {
        video,
        segments,
        detections,
    })
}
