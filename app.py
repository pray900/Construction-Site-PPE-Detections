import os
import time
import tempfile
import numpy as np
import cv2
import streamlit as st


# Weights sit in a folder next to this file, so the app runs from any clone
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")


MODEL_CONFIG = {
    "YOLOv5n": {
        "path": os.path.join(WEIGHTS_DIR, "yolov5n_best.pt"),
        "loader": "YOLO",
    },
    "YOLOv8s": {
        "path": os.path.join(WEIGHTS_DIR, "yolov8s_best.pt"),
        "loader": "YOLO",
    },
    "RT-DETR": {
        "path": os.path.join(WEIGHTS_DIR, "rtdetr_best.pt"),
        "loader": "RTDETR",
    },
}

CLASS_NAMES = {0: "helmet", 1: "vest", 2: "head"}

# Colours for drawing
COLOURS_BGR = {
    "helmet": (0, 200, 0),        # green
    "vest": (200, 150, 0),        # blue-ish
    "head_safe": (0, 165, 255),   # orange (head WITH helmet)
    "head_unsafe": (0, 0, 220),   # red (head WITHOUT helmet)
    "torso_zone": (180, 180, 0),  # cyan dashed 
}
# Same colours in RGB for Streamlit display text
RISK_COLOURS = {
    "LOW": "#2ecc71",        
    "MODERATE": "#f1c40f",  
    "HIGH": "#e67e22",      
    "CRITICAL": "#e74c3c",   
}

# Inference settings
DEFAULT_IMGSZ = 832
DEFAULT_CONF = 0.25
DEFAULT_FRAME_STRIDE = 1        # video: analyse every Nth frame


# MODEL LOADING (cached so it doesn't reload on every interaction)
@st.cache_resource
def load_model(model_name):    #Load a model by name. Returns the model object or None if unavailable
    cfg = MODEL_CONFIG.get(model_name)
    if cfg is None:
        return None

    weight_path = cfg["path"]
    if not os.path.exists(weight_path):
        return None

    # Import the right class from Ultralytics to read our model
    if cfg["loader"] == "RTDETR":
        from ultralytics import RTDETR
        model = RTDETR(weight_path)
    else:
        from ultralytics import YOLO
        model = YOLO(weight_path)
    return model


# DETECTION — run the model on BGR image
def run_detection(model, frame_bgr, imgsz, conf):
    results = model(frame_bgr, imgsz=imgsz, conf=conf, verbose=False)[0]

    detections = []
    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()     
        classes = results.boxes.cls.cpu().numpy()      
        confs = results.boxes.conf.cpu().numpy()     
        for i in range(len(boxes)):
            cid = int(classes[i])
            detections.append({
                "box": boxes[i].tolist(),             
                "class_id": cid,
                "class_name": CLASS_NAMES.get(cid, f"class_{cid}"),
                "confidence": float(confs[i]),
            })
    return detections

# HEAD-ANCHORED VIOLATION LOGIC
def box_centre(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def boxes_overlap(a, b):
    if a[0] >= b[2] or b[0] >= a[2]:    # no horizontal overlap
        return False
    if a[1] >= b[3] or b[1] >= a[3]:    # no vertical overlap
        return False
    return True


def point_in_box(px, py, box):
    return box[0] <= px <= box[2] and box[1] <= py <= box[3]


def analyse_violations(detections):
    # Separate detections by class
    heads = [d for d in detections if d["class_name"] == "head"]
    helmets = [d for d in detections if d["class_name"] == "helmet"]
    vests = [d for d in detections if d["class_name"] == "vest"]

    workers = []
    matched_helmet_indices = set()   # track which helmets were linked to a head

    # ========================== PASS 1: HEAD-BASED WORKERS ==========================
    for head in heads:
        hb = head["box"]                       # [x1, y1, x2, y2]
        h_cx = (hb[0] + hb[2]) / 2            # head centre x
        h_w = hb[2] - hb[0]                   # head width
        h_h = hb[3] - hb[1]                   # head height

        # --- 1. HELMET ZONE: above the head ---
        hz_w = h_w * 1.5
        hz_h = h_h * 1.5
        helmet_zone = [
            h_cx - hz_w / 2,                  # x1
            hb[1] - hz_h,                     # y1 (above head top)
            h_cx + hz_w / 2,                  # x2
            hb[1],                             # y2 (head top)
        ]

        # Check each helmet detection against this zone
        has_helmet = False
        for hi, helmet in enumerate(helmets):
            hel_box = helmet["box"]
            hel_cx, hel_cy = box_centre(hel_box)
            if boxes_overlap(helmet_zone, hel_box) or \
               point_in_box(hel_cx, hel_cy, helmet_zone):
                has_helmet = True
                matched_helmet_indices.add(hi)     # mark this helmet as used
                break

        # --- 2. TORSO ZONE: below the head ---
        tz_w = h_w * 2.5
        tz_h = h_h * 4.5
        torso_zone = [
            h_cx - tz_w / 2,
            hb[3],                             # starts at head bottom
            h_cx + tz_w / 2,
            hb[3] + tz_h,
        ]

        has_vest = False
        for vest in vests:
            v_box = vest["box"]
            v_cx, v_cy = box_centre(v_box)
            if boxes_overlap(torso_zone, v_box) or \
               point_in_box(v_cx, v_cy, torso_zone):
                has_vest = True
                break

        # --- 3. Worker status ---
        missing = []
        if not has_helmet:
            missing.append("helmet")
        if not has_vest:
            missing.append("vest")

        if has_helmet and has_vest:
            status = "COMPLIANT"
            score = 100
        elif has_helmet or has_vest:
            status = "PARTIAL"
            score = 50
        else:
            status = "NON-COMPLIANT"
            score = 0

        workers.append({
            "head_box": hb,
            "has_helmet": has_helmet,
            "has_vest": has_vest,
            "status": status,
            "missing": missing,
            "score": score,
            "helmet_zone": helmet_zone,
            "torso_zone": torso_zone,
        })

    # =================== PASS 2: HELMET-ONLY WORKERS (no visible head) ===================
    # A helmet with no matching head means the helmet occludes the head.
    # This worker IS wearing a helmet, but we still need to check for a vest.
    for hi, helmet in enumerate(helmets):
        if hi in matched_helmet_indices:
            continue                           # already linked to a head in Pass 1

        hel_box = helmet["box"]                # use helmet box as the anchor
        hel_cx = (hel_box[0] + hel_box[2]) / 2
        hel_w = hel_box[2] - hel_box[0]
        hel_h = hel_box[3] - hel_box[1]

        # Torso zone: below the helmet box (same multipliers as Pass 1)
        tz_w = hel_w * 2.5
        tz_h = hel_h * 4.5
        torso_zone = [
            hel_cx - tz_w / 2,
            hel_box[3],                        # starts at helmet bottom
            hel_cx + tz_w / 2,
            hel_box[3] + tz_h,
        ]

        has_vest = False
        for vest in vests:
            v_box = vest["box"]
            v_cx, v_cy = box_centre(v_box)
            if boxes_overlap(torso_zone, v_box) or \
               point_in_box(v_cx, v_cy, torso_zone):
                has_vest = True
                break

        missing = []
        if not has_vest:
            missing.append("vest")

        if has_vest:
            status = "COMPLIANT"
            score = 100
        else:
            status = "PARTIAL"
            score = 50

        workers.append({
            "head_box": hel_box,               # use helmet box as proxy for head
            "has_helmet": True,                # this worker definitely has a helmet
            "has_vest": has_vest,
            "status": status,
            "missing": missing,
            "score": score,
            "helmet_zone": None,               # no helmet zone needed (helmet IS the anchor)
            "torso_zone": torso_zone,
        })

    return workers


def compute_safety_score(workers):
    if not workers:
        return 100.0, "LOW"                    # no workers = no violations

    avg = sum(w["score"] for w in workers) / len(workers)

    if avg >= 90:
        risk = "LOW"
    elif avg >= 70:
        risk = "MODERATE"
    elif avg >= 40:
        risk = "HIGH"
    else:
        risk = "CRITICAL"

    return round(avg, 1), risk


# DRAWING — annotate a BGR frame with detections + violation overlays
def draw_dashed_rect(img, pt1, pt2, colour, thickness=1, dash_len=10):
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    # Top edge
    for x in range(x1, x2, dash_len * 2):
        cv2.line(img, (x, y1), (min(x + dash_len, x2), y1), colour, thickness)
    # Bottom edge
    for x in range(x1, x2, dash_len * 2):
        cv2.line(img, (x, y2), (min(x + dash_len, x2), y2), colour, thickness)
    # Left edge
    for y in range(y1, y2, dash_len * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash_len, y2)), colour, thickness)
    # Right edge
    for y in range(y1, y2, dash_len * 2):
        cv2.line(img, (x2, y), (x2, min(y + dash_len, y2)), colour, thickness)


def annotate_frame(frame_bgr, detections, workers):

    img = frame_bgr.copy()
    h_img, w_img = img.shape[:2]

    # Build a set of "safe" heads (those with helmets) for colouring
    safe_heads = set()
    for w in workers:
        if w["has_helmet"]:
            safe_heads.add(tuple(w["head_box"]))

    # --- Draw synthesised zones (faint, behind the main boxes) ---
    for w in workers:
        # Torso zone — dashed cyan rectangle
        tz = w["torso_zone"]
        draw_dashed_rect(
            img,
            (max(0, tz[0]), max(0, tz[1])),
            (min(w_img, tz[2]), min(h_img, tz[3])),
            COLOURS_BGR["torso_zone"],
            thickness=1, dash_len=8,
        )

    # --- Draw detection boxes ---
    for det in detections:
        box = det["box"]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cls = det["class_name"]
        conf = det["confidence"]

        # Choose colour based on class and safety status
        if cls == "helmet":
            colour = COLOURS_BGR["helmet"]
        elif cls == "vest":
            colour = COLOURS_BGR["vest"]
        elif cls == "head":
            # Head colour depends on whether this head has a helmet
            if tuple(box) in safe_heads:
                colour = COLOURS_BGR["head_safe"]    # orange = has helmet
            else:
                colour = COLOURS_BGR["head_unsafe"]  # red = no helmet
        else:
            colour = (200, 200, 200)                 # grey fallback

        # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)

        # Label text
        label = f"{cls} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Label background
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    # --- Draw worker status labels near each head ---
    for i, w in enumerate(workers):
        hb = w["head_box"]
        cx = int((hb[0] + hb[2]) / 2)
        y_label = int(hb[3]) + 15                    # just below head box

        status_text = w["status"]
        if w["missing"]:
            status_text += f" (no {', '.join(w['missing'])})"

        # Background for status text
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(status_text, font, 0.45, 1)
        if w["status"] == "COMPLIANT":
            bg_colour = (0, 160, 0)
        elif w["status"] == "PARTIAL":
            bg_colour = (0, 165, 255)
        else:
            bg_colour = (0, 0, 200)

        tx = max(0, cx - tw // 2)
        cv2.rectangle(img, (tx - 2, y_label - th - 2),
                      (tx + tw + 2, y_label + 4), bg_colour, -1)
        cv2.putText(img, status_text, (tx, y_label), font, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return img


# DISPLAY RESULTS — show the results panel in Streamlit
def display_results(workers, safety_score, risk_level,
                    detections=None, inference_ms=None):

    # Safety score — large, colour-coded
    colour = RISK_COLOURS.get(risk_level, "#ffffff")
    st.markdown(
        f"<h1 style='text-align:center; color:{colour};'>{safety_score}</h1>"
        f"<h3 style='text-align:center; color:{colour};'>{risk_level} RISK</h3>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Summary counts
    total = len(workers)
    compliant = sum(1 for w in workers if w["status"] == "COMPLIANT")
    partial = sum(1 for w in workers if w["status"] == "PARTIAL")
    non_compliant = sum(1 for w in workers if w["status"] == "NON-COMPLIANT")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Workers", total)
    col2.metric("Compliant", compliant)
    col3.metric("Non-Compliant", partial + non_compliant)

    st.markdown("---")

    # Per-worker breakdown
    if workers:
        st.markdown("**Per-Worker Breakdown:**")
        for i, w in enumerate(workers):
            helmet_icon = "✅" if w["has_helmet"] else "❌"
            vest_icon = "✅" if w["has_vest"] else "❌"
            status_icon = {"COMPLIANT": "🟢", "PARTIAL": "🟡",
                           "NON-COMPLIANT": "🔴"}.get(w["status"], "⚪")
            st.markdown(
                f"{status_icon} **Worker {i+1}:** "
                f"Helmet {helmet_icon}  Vest {vest_icon} — "
                f"*{w['status']}*"
            )
    else:
        st.info("No workers detected in this image.")

    st.markdown("---")

    # 1. Detection Summary — raw model output counts
    if detections is not None:
        n_helmets = sum(1 for d in detections if d["class_name"] == "helmet")
        n_vests = sum(1 for d in detections if d["class_name"] == "vest")
        n_heads = sum(1 for d in detections if d["class_name"] == "head")
        st.markdown("**Detection Summary:**")
        st.markdown(
            f"🪖 Helmets detected: **{n_helmets}** &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"🦺 Vests detected: **{n_vests}** &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"👤 Heads detected: **{n_heads}**"
        )

    # 2. Compliance Rate
    if total > 0:
        rate = (compliant / total) * 100
        st.markdown(
            f"**Compliance Rate:** {compliant}/{total} workers fully "
            f"compliant (**{rate:.0f}%**)"
        )
    elif detections is not None and len(detections) == 0:
        st.markdown("**Compliance Rate:** No detections in this image.")

    # 3. Specific Violations
    if total > 0:
        missing_helmet = sum(1 for w in workers if not w["has_helmet"])
        missing_vest = sum(1 for w in workers if not w["has_vest"])
        if missing_helmet > 0 or missing_vest > 0:
            st.markdown("**Violations Found:**")
            if missing_helmet > 0:
                st.markdown(f"⚠️ {missing_helmet} worker(s) missing **helmet**")
            if missing_vest > 0:
                st.markdown(f"⚠️ {missing_vest} worker(s) missing **vest**")
        else:
            st.markdown("✅ **No violations — all workers fully compliant.**")

    # 4. Inference Time
    if inference_ms is not None:
        st.markdown("---")
        st.markdown(f"⏱️ **Inference Time:** {inference_ms:.1f} ms")


# PROCESS A SINGLE FRAME — detection + violation analysis + annotation
def process_frame(model, frame_bgr, imgsz, conf):

    t0 = time.time()
    detections = run_detection(model, frame_bgr, imgsz, conf)
    inference_ms = (time.time() - t0) * 1000     # milliseconds

    workers = analyse_violations(detections)
    safety_score, risk_level = compute_safety_score(workers)
    annotated = annotate_frame(frame_bgr, detections, workers)
    return annotated, detections, workers, safety_score, risk_level, inference_ms


# INPUT MODE HANDLERS
def handle_image_upload(model, imgsz, conf):

    uploaded = st.file_uploader(
        "Upload a construction site image",
        type=["jpg", "jpeg", "png", "bmp"],
        key="image_uploader",
    )
    if uploaded is not None:
        # Read the uploaded image into a BGR numpy array
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            st.error("Could not read the uploaded image. Please try another file.")
            return

        with st.spinner("Running detection..."):
            annotated, detections, workers, score, risk, inf_ms = process_frame(
                model, frame_bgr, imgsz, conf
            )

        # Display: image on the left, results on the right
        col_img, col_res = st.columns([3, 2])
        with col_img:
            # Convert BGR -> RGB for Streamlit display
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="Detection Results", use_container_width=True)
        with col_res:
            display_results(workers, score, risk, detections, inf_ms)


def remove_temp_files(*paths):
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass


def handle_video_upload(model, imgsz, conf, stride=DEFAULT_FRAME_STRIDE):

    uploaded = st.file_uploader(
        "Upload a construction site video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
    )
    if uploaded is not None:
        # Save the uploaded video to a temp file (OpenCV needs a file path).
        # Close the handle so Windows lets us delete the file afterwards.
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile_in.write(uploaded.read())
        tfile_in.close()

        # Open the video
        cap = cv2.VideoCapture(tfile_in.name)
        if not cap.isOpened():
            st.error("Could not open the uploaded video.")
            remove_temp_files(tfile_in.name)
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Prepare output video writer. Browsers can only play H.264 (avc1), so
        # try that first and fall back to mp4v if this OpenCV build lacks it.
        # Dropping the FPS by the stride keeps the output the same duration.
        tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile_out.close()
        out_fps = max(fps / stride, 1.0)
        writer = cv2.VideoWriter(tfile_out.name,
                                 cv2.VideoWriter_fourcc(*"avc1"),
                                 out_fps, (width, height))
        if not writer.isOpened():
            writer = cv2.VideoWriter(tfile_out.name,
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     out_fps, (width, height))

        if stride > 1:
            st.info(f"Processing every {stride}th frame of {total_frames} "
                    f"at {fps:.1f} FPS...")
        else:
            st.info(f"Processing {total_frames} frames at {fps:.1f} FPS...")
        progress = st.progress(0)
        frame_display = st.empty()           # placeholder for live preview

        frame_idx = 0        # frames actually analysed
        read_idx = 0         # frames read from the source video

        # Last frame results (overwritten each frame)
        last_detections = []
        last_workers = []
        last_score =100.0
        last_risk = "LOW"
        last_inf_ms = 0.0

        # Per-frame stats for video summary
        frame_worker_counts = []
        frame_scores = []
        frame_inf_times = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames between strides — they cost nothing but a read
            if read_idx % stride != 0:
                read_idx += 1
                continue

            annotated, dets, workers, score, risk, inf_ms = process_frame(
                model, frame, imgsz, conf
            )
            writer.write(annotated)

            # Overwrite last frame results
            last_detections = dets
            last_workers = workers
            last_score = score
            last_risk = risk
            last_inf_ms = inf_ms

            # Record per-frame stats
            frame_worker_counts.append(len(workers))
            frame_scores.append(score)
            frame_inf_times.append(inf_ms)

            # Show every 10th analysed frame as a preview
            if frame_idx % 10 == 0:
                frame_display.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"Frame {read_idx}/{total_frames}",
                    use_container_width=True,
                )

            frame_idx += 1
            read_idx += 1
            if total_frames > 0:
                progress.progress(min(read_idx / total_frames, 1.0))

        cap.release()
        writer.release()
        progress.empty()
        frame_display.empty()

        # Compute video-level averages
        n_frames = max(frame_idx, 1)
        avg_workers = sum(frame_worker_counts) / n_frames
        avg_score = sum(frame_scores) / n_frames
        avg_inf_ms = sum(frame_inf_times) / n_frames

        # Determine risk level for the average score
        if avg_score >= 90:
            avg_risk = "LOW"
        elif avg_score >= 70:
            avg_risk = "MODERATE"
        elif avg_score >= 40:
            avg_risk = "HIGH"
        else:
            avg_risk = "CRITICAL"

        # Read the annotated video once — we serve the bytes to both widgets so
        # nothing depends on the temp file still being on disk afterwards.
        with open(tfile_out.name, "rb") as f:
            video_bytes = f.read()

        col_vid, col_res = st.columns([3, 2])
        with col_vid:
            st.download_button(
                label="📥 Download Annotated Video",
                data=video_bytes,
                file_name="annotated_output.mp4",
                mime="video/mp4",
            )
            st.video(video_bytes)
        with col_res:
            # Section A — Last Frame Analysis
            st.markdown("**Last Frame Analysis**")
            display_results(last_workers, last_score, last_risk,
                            last_detections, last_inf_ms)

            # Section B — Video Summary
            st.markdown("---")
            st.markdown("**Video Summary**")
            st.markdown(f"Avg workers per frame: **{avg_workers:.1f}**")
            st.markdown(f"Avg safety score: **{avg_score:.1f}** ({avg_risk})")
            st.markdown(f"Avg inference time: **{avg_inf_ms:.1f}** ms per frame")
            st.markdown(
                f"Frames analysed: **{frame_idx}** of **{read_idx}** read"
                + (f" (every {stride}th frame)" if stride > 1 else "")
            )

        # Clean up temp files
        remove_temp_files(tfile_in.name, tfile_out.name)



def handle_webcam_capture(model, imgsz, conf):

    st.info("Click the camera button below to capture a snapshot from your webcam.")
    camera_image = st.camera_input("Capture a snapshot")

    if camera_image is not None:
        # Read the captured image
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            st.error("Could not read the webcam image.")
            return

        with st.spinner("Running detection..."):
            annotated, detections, workers, score, risk, inf_ms = process_frame(
                model, frame_bgr, imgsz, conf
            )

        col_img, col_res = st.columns([3, 2])
        with col_img:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="Detection Results", use_container_width=True)
        with col_res:
            display_results(workers, score, risk, detections, inf_ms)


# MAIN APP
def main():
    st.set_page_config(
        page_title="Construction Site Safety Monitor",
        page_icon="🏗️",
        layout="wide",
    )

    # ---- SIDEBAR ----
    st.sidebar.title("🏗️ Safety Monitor")
    st.sidebar.markdown("---")

    # Model selector
    model_names = list(MODEL_CONFIG.keys())
    selected_model = st.sidebar.selectbox("Select Model", model_names)

    # Check if the selected model's weight file exists
    weight_path = MODEL_CONFIG[selected_model]["path"]
    model_available = os.path.exists(weight_path)

    if not model_available:
        st.sidebar.warning(
            f"⚠️ Weights not found for **{selected_model}**.\n\n"
            f"Expected at:\n`{weight_path}`\n\n"
            f"Run `git lfs pull` to fetch the weights, or download them "
            f"from the repository releases into the `weights/` folder."
        )

    # Input mode selector
    input_mode = st.sidebar.radio(
        "Input Mode",
        ["📷 Image Upload", "🎬 Video Upload", "📸 Webcam Capture"],
    )

    # Confidence threshold
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.10, max_value=0.90, value=DEFAULT_CONF, step=0.05,
    )

    # Video runs one inference per frame, so a stride keeps long clips usable
    frame_stride = DEFAULT_FRAME_STRIDE
    if input_mode == "🎬 Video Upload":
        frame_stride = st.sidebar.slider(
            "Analyse every Nth frame",
            min_value=1, max_value=10, value=DEFAULT_FRAME_STRIDE, step=1,
            help="Higher values process fewer frames and finish faster.",
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Classes:** helmet, vest, head\n\n"
        "**Violation Logic:** Head-anchored — each detected head is "
        "checked for helmet above and vest below.\n\n"
        "**Safety Score:** COMPLIANT=100, PARTIAL=50, NON-COMPLIANT=0. "
        "Site score = average."
    )

    # ---- MAIN AREA ----
    st.title("🏗️ Construction Site Safety Detection")
    st.markdown(
        "Upload construction site images or video to detect PPE compliance. "
        "The system identifies workers (heads), helmets, and vests, then "
        "analyses safety violations."
    )

    if not model_available:
        st.error(
            f"**{selected_model}** weights not found. "
            f"Please download the `best.pt` file to:\n\n"
            f"`{weight_path}`"
        )
        return

    # Load the selected model (cached — only loads once per model)
    model = load_model(selected_model)
    if model is None:
        st.error(f"Failed to load model **{selected_model}**.")
        return

    st.success(f"Model loaded: **{selected_model}**")

    # Route to the selected input mode
    if input_mode == "📷 Image Upload":
        handle_image_upload(model, DEFAULT_IMGSZ, conf_threshold)
    elif input_mode == "🎬 Video Upload":
        handle_video_upload(model, DEFAULT_IMGSZ, conf_threshold, frame_stride)
    elif input_mode == "📸 Webcam Capture":
        handle_webcam_capture(model, DEFAULT_IMGSZ, conf_threshold)


if __name__ == "__main__":
    main()
