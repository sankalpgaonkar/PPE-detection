import streamlit as st
import tempfile
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import base64
from datetime import datetime

# ─────────────────────────────────────────
# GLOBAL LOGO SETUP
# ─────────────────────────────────────────
logo_path = "logo.png"
logo_small_path = "logo_small.png"

if os.path.exists(logo_path):
    page_icon_img = Image.open(logo_path)
else:
    page_icon_img = "🦺"

if os.path.exists(logo_small_path):
    with open(logo_small_path, "rb") as image_file:
        encoded_logo = base64.b64encode(image_file.read()).decode()
    logo_html = f'<img src="data:image/png;base64,{encoded_logo}" style="height: 50px; object-fit: contain;">'
    logo_html_small = f'<img src="data:image/png;base64,{encoded_logo}" style="height: 24px; vertical-align: middle;">'
else:
    logo_html = '<span style="font-size:2em;">🦺</span>'
    logo_html_small = '👷'

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
PPE_HEX_COLORS = {
    'Boots':          '#FFA500',
    'Ear-protection': '#9B59B6',
    'Glass':          '#00BCD4',
    'Glove':          '#F1C40F',
    'Helmet':         '#2ECC71',
    'Mask':           '#FF69B4',
    'Person':         '#E74C3C',
    'Vest':           '#3498DB',
}

PPE_COLORS_BGR = {
    'Boots':          (0,   165, 255),
    'Ear-protection': (128,   0, 128),
    'Glass':          (255, 255,   0),
    'Glove':          (0,   255, 255),
    'Helmet':         (0,   255,   0),
    'Mask':           (203, 192, 255),
    'Person':         (0,     0, 255),
    'Vest':           (255,   0,   0),
}

CONF_THRESHOLD = 0.50

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("best.pt")


# ─────────────────────────────────────────
# DETECTION HELPERS
# ─────────────────────────────────────────
def draw_custom_boxes(frame, results, model_names):
    out = frame.copy()
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id     = int(box.cls[0].item())
        conf       = float(box.conf[0].item())
        class_name = model_names[cls_id]
        if class_name not in PPE_COLORS_BGR:
            continue
        color = PPE_COLORS_BGR[class_name]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_color = (0, 0, 0) if class_name in ['Glass', 'Glove', 'Helmet'] else (255, 255, 255)
        y_bg = max(0, y1 - 20)
        cv2.rectangle(out, (x1, y_bg), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 5 if y1 > 20 else y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    return out


def get_current_data(results, model_names):
    counts = {}
    confs = {}
    if len(results.boxes) > 0:
        for cls_id, conf in zip(results.boxes.cls.cpu().numpy(), results.boxes.conf.cpu().numpy()):
            name = model_names[int(cls_id)]
            if name in PPE_HEX_COLORS:
                counts[name] = counts.get(name, 0) + 1
                confs[name] = max(confs.get(name, 0), float(conf * 100))
    return counts, confs


def update_max_counts(current_counts, max_counts):
    for name, cnt in current_counts.items():
        if cnt > max_counts.get(name, 0):
            max_counts[name] = cnt
    return max_counts


# ─────────────────────────────────────────
# LIVE DASHBOARD HTML
# ─────────────────────────────────────────
def build_dashboard_html(current_counts, current_confs, max_counts):
    person_now = current_counts.get("Person", 0)
    person_max = max_counts.get("Person", 0)
    ppe_classes = [c for c in PPE_HEX_COLORS if c != "Person"]

    # Live fluctuating bars - based on detection confidence
    bars_html = ""
    for ppe in sorted(ppe_classes):
        if current_counts.get(ppe, 0) == 0:
            continue
        conf = current_confs.get(ppe, 0)
        color = PPE_HEX_COLORS[ppe]
        pct = int(conf)
        bars_html += f"""
        <div style="flex:1;min-width:110px;max-width:160px;">
          <div style="display:flex;justify-content:space-between;font-size:0.75em;
                      font-weight:600;margin-bottom:4px;color:{color};">
            <span>{ppe}</span><span>{pct}%</span>
          </div>
          <div style="height:8px;border-radius:4px;background:rgba(128,128,128,0.25);overflow:hidden;">
            <div style="height:100%;width:{pct}%;background:{color};border-radius:4px;"></div>
          </div>
        </div>"""

    if not bars_html:
        bars_html = "<span style='color:#aaa;font-size:0.75em;'>No objects in frame yet…</span>"

    top_right_badge = f"""
    <div style="position:fixed; top:20px; right:20px; z-index:9999;
                background:rgba(20,20,20,0.85); color:white; padding:10px 18px;
                border-radius:12px; border:1px solid rgba(255,255,255,0.15);
                box-shadow:0 6px 12px rgba(0,0,0,0.4); font-family:sans-serif;
                display:flex; align-items:center; gap:12px; backdrop-filter: blur(4px);">
        {logo_html_small}
        <div style="text-align:right;">
            <div style="font-size:0.7em; text-transform:uppercase; letter-spacing:1px; color:#aaa; margin-bottom:2px;">Live Workers</div>
            <div style="font-size:1.6em; font-weight:800; color:#2ECC71; line-height:1;">{person_now}</div>
        </div>
    </div>
    """

    indicator_block = f"""
    <div style="display:flex;gap:14px;flex-wrap:wrap;align-items:center;
                padding:8px 12px;border-radius:8px;
                background:rgba(100,100,100,0.08);margin-bottom:14px;
                border:1px solid rgba(128,128,128,0.15);">
      <div style="flex:1;display:flex;gap:12px;flex-wrap:wrap;">
        {bars_html}
      </div>
    </div>"""

    # Summary table using current_counts
    rows = ""
    for ppe in sorted(ppe_classes):
        with_ppe    = current_counts.get(ppe, 0)
        if with_ppe == 0:
            continue
        without_ppe = max(0, person_now - with_ppe)
        color       = PPE_HEX_COLORS[ppe]
        dot         = f"<span style='color:{color};margin-right:6px;'>●</span>"
        rows += f"""
        <tr style="border-bottom:1px solid rgba(128,128,128,0.1);">
          <td style="padding:7px 12px;">{dot}{ppe}</td>
          <td style="padding:7px 12px;text-align:center;color:#2ECC71;font-weight:700;">{with_ppe}</td>
          <td style="padding:7px 12px;text-align:center;color:#E74C3C;font-weight:700;">{without_ppe}</td>
        </tr>"""

    if not rows:
        rows = """
        <tr>
          <td colspan="3" style="padding:15px;text-align:center;color:#aaa;font-size:0.9em;font-style:italic;">
            No equipment detected yet.
          </td>
        </tr>"""

    table_block = f"""
    <div style="overflow-x:auto;border-radius:8px;
                border:1px solid rgba(128,128,128,0.15);">
      <table style="width:100%;border-collapse:collapse;font-size:0.86em;font-family:sans-serif;">
        <thead>
          <tr style="background:rgba(100,100,100,0.12);font-size:0.78em;
                     letter-spacing:0.06em;text-transform:uppercase;">
            <th style="padding:9px 12px;text-align:left;font-weight:600;">Equipment</th>
            <th style="padding:9px 12px;text-align:center;font-weight:600;color:#2ECC71;">With ✅</th>
            <th style="padding:9px 12px;text-align:center;font-weight:600;color:#E74C3C;">Without ❌</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""

    return f"""
    {top_right_badge}
    <div style="width:100%;font-family:sans-serif;padding:2px 0;">
      {indicator_block}
      {table_block}
    </div>"""


# ─────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────
def generate_report(max_counts, input_type):
    report  = "# PPE Safety Detection Report\n\n"
    report += f"**Input Type:** {input_type}\n"
    report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    person  = max_counts.get("Person", 0)
    report += f"## Peak Workers Detected: {person}\n\n"
    report += "## Compliance Breakdown\n\n"
    report += "| Equipment | With | Without | Compliance % |\n"
    report += "|-----------|------|---------|--------------|\n"
    for ppe in sorted(PPE_HEX_COLORS):
        if ppe == "Person":
            continue
        w   = max_counts.get(ppe, 0)
        if w == 0:
            continue
        wo  = max(0, person - w)
        pct = int((w / person) * 100) if person > 0 else 0
        report += f"| {ppe} | {w} | {wo} | {pct}% |\n"
    report += "\n_Counts represent the peak value seen in any single frame._\n"
    return report


# ─────────────────────────────────────────
# IMAGE DETECTION
# ─────────────────────────────────────────
def detect_image(image, model, filename="detected_image"):
    bgr     = np.array(image)[:, :, ::-1].copy()
    results = model(bgr, conf=CONF_THRESHOLD, verbose=False)
    annotated = draw_custom_boxes(bgr, results[0], model.names)
    current, cur_confs = get_current_data(results[0], model.names)
    max_c     = update_max_counts(current, {})
    rgb       = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", f"{filename}_detected.jpg")
    cv2.imwrite(out_path, annotated)
    return rgb, out_path, current, cur_confs, max_c


# ─────────────────────────────────────────
# VIDEO DETECTION
# ─────────────────────────────────────────
def detect_video(video_path, model, frame_ph, stats_ph):
    cap      = cv2.VideoCapture(video_path)
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs("output", exist_ok=True)
    base     = os.path.basename(video_path).split('.')[0]
    out_path = os.path.join("output", f"{base}_detected.mp4")
    out      = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    max_c    = {}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results   = model(frame, conf=CONF_THRESHOLD, verbose=False)
        annotated = draw_custom_boxes(frame, results[0], model.names)
        current, cur_confs = get_current_data(results[0], model.names)
        max_c     = update_max_counts(current, max_c)
        frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                       channels="RGB", use_container_width=True)
        stats_ph.html(build_dashboard_html(current, cur_confs, max_c))
        out.write(annotated)
    cap.release()
    out.release()
    return out_path, max_c


# ─────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────
st.set_page_config(layout="wide", page_title="PPE Safety Monitor", page_icon=page_icon_img)

st.html("""
<style>
  [data-testid="stHeader"]    { visibility:hidden; height:0; min-height:0; }
  [data-testid="stToolbar"]   { visibility:hidden; }
  #MainMenu                   { visibility:hidden; }
  footer                      { visibility:hidden; height:0; min-height:0; }
  [data-testid="stStatusWidget"] { visibility:hidden; }
  [data-testid="stDecoration"]   { display:none; }
  .viewerBadge_container__1QSob,
  .viewerBadge_link__1S137,
  [class*="viewerBadge_container"],
  [data-testid="manage-app-button"] { display:none !important; }
  .block-container             { padding-top:0.6rem !important;
                                 padding-bottom:0 !important;
                                 max-width:100% !important; }
  [data-testid="stImage"] img { max-height:80vh !important;
                                 width:100% !important;
                                 object-fit:contain !important; }
</style>
""")


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.markdown(f"## {logo_html_small} PPE Monitor", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <div style="margin-bottom: 12px; padding: 6px 10px; background: rgba(255,255,255,0.05); border-radius: 6px; border: 1px solid rgba(128,128,128,0.2); font-size: 0.8em; color: #888;">
        Hosted by <b style="color: #2ECC71;">Sankalp Gaonkar</b>
    </div>
    """,
    unsafe_allow_html=True,
)
input_type = st.sidebar.radio("Source", ["Image", "Video", "Webcam"], label_visibility="collapsed")
model = load_model()

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="font-size:0.75em;color:#888;text-align:center;line-height:1.4;">
      PPE Safety Monitor<br>
      Hosted by <b><a href="https://github.com/sankalpgaonkar" target="_blank" style="color:#2ECC71;text-decoration:none;">Sankalp Gaonkar</a></b>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────
# ─────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────

st.html(f"""
<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:15px;margin-bottom:15px;">
  <div style="display:flex;align-items:center;gap:15px;">
    {logo_html}
    <div>
      <div style="font-size:1.35em;font-weight:700;line-height:1.2;">PPE Safety Detection Dashboard</div>
      <div style="font-size:0.8em;color:#888;">Real-time Personal Protective Equipment monitoring</div>
    </div>
  </div>
  <div style="font-size:0.85em;color:#aaa;background:rgba(255,255,255,0.06);padding:6px 12px;border-radius:8px;border:1px solid rgba(128,128,128,0.2);">
    Hosted by <b style="color:#2ECC71;">Sankalp Gaonkar</b>
  </div>
</div>
""")


# ─────────────────────────────────────────
# IMAGE
# ─────────────────────────────────────────
if input_type == "Image":
    uploaded = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        fname = os.path.splitext(uploaded.name)[0]
        with st.spinner("Detecting…"):
            rgb, out_path, current, cur_confs, max_c = detect_image(image, model, filename=fname)
        frame_ph = st.empty()
        stats_ph = st.empty()
        frame_ph.image(rgb, use_container_width=True)
        stats_ph.html(build_dashboard_html(current, cur_confs, max_c))
        st.divider()
        report = generate_report(max_c, "Image")
        
        st.markdown("### 📊 Final Detection Report")
        st.markdown(report)
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇ Download Report", report,
                               file_name=f"{fname}_report.md", mime="text/markdown")
        with c2:
            with open(out_path, "rb") as f:
                st.download_button("⬇ Download Detected Image", f,
                                   file_name=os.path.basename(out_path))
    else:
        st.info("📂 Upload an image from the sidebar to begin detection.")


# ─────────────────────────────────────────
# VIDEO
# ─────────────────────────────────────────
elif input_type == "Video":
    uploaded = st.sidebar.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    if uploaded:
        frame_ph = st.empty()
        stats_ph = st.empty()
        with st.spinner("Running detection on video…"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded.read())
                vpath = tmp.name
            out_path, max_c = detect_video(vpath, model, frame_ph, stats_ph)
        st.success("✅ Video processing complete!")
        st.divider()
        report = generate_report(max_c, "Video")
        
        st.markdown("### 📊 Final Detection Report")
        st.markdown(report)
        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇ Download Report", report,
                               file_name="video_report.md", mime="text/markdown")
        with c2:
            with open(out_path, "rb") as f:
                st.download_button("⬇ Download Processed Video", f,
                                   file_name=os.path.basename(out_path))
    else:
        st.info("📂 Upload a video from the sidebar to begin detection.")


# ─────────────────────────────────────────
# WEBCAM
# ─────────────────────────────────────────
elif input_type == "Webcam":
    # Initialise session state
    if "webcam_on"   not in st.session_state: st.session_state.webcam_on   = False
    if "max_counts"  not in st.session_state: st.session_state.max_counts  = {}
    if "webcam_done" not in st.session_state: st.session_state.webcam_done = False

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("▶ Start Webcam", use_container_width=True):
            st.session_state.webcam_on   = True
            st.session_state.webcam_done = False
            st.session_state.max_counts  = {}
            st.rerun()
    with col_b:
        if st.button("⏹ Stop Webcam", use_container_width=True):
            st.session_state.webcam_on   = False
            st.session_state.webcam_done = True
            st.rerun()

    frame_ph = st.empty()
    stats_ph = st.empty()

    if st.session_state.webcam_on:
        # Open webcam, read ONE frame, display it, then immediately rerun.
        # This ensures the Stop button's rerun can interrupt the loop.
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Could not open webcam. Please check camera permissions.")
            st.session_state.webcam_on = False
        else:
            ret, frame = cap.read()
            cap.release()          # release right away; re-open next rerun
            if ret:
                results   = model(frame, conf=CONF_THRESHOLD, verbose=False)
                annotated = draw_custom_boxes(frame, results[0], model.names)
                current, cur_confs = get_current_data(results[0], model.names)
                st.session_state.max_counts = update_max_counts(
                    current, st.session_state.max_counts)
                frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                               channels="RGB", use_container_width=True)
                stats_ph.html(
                    build_dashboard_html(current, cur_confs, st.session_state.max_counts))
                # Immediately trigger next frame
                st.rerun()
            else:
                st.error("❌ Failed to read from webcam.")
                st.session_state.webcam_on = False

    elif st.session_state.webcam_done and st.session_state.max_counts:
        st.success("✅ Webcam session ended.")
        stats_ph.html(
            build_dashboard_html({}, {}, st.session_state.max_counts))
        st.divider()
        report = generate_report(st.session_state.max_counts, "Webcam")
        
        st.markdown("### 📊 Final Detection Report")
        st.markdown(report)
        st.divider()

        st.download_button("⬇ Download Report", report,
                           file_name="webcam_report.md", mime="text/markdown")
    else:
        st.info("Press **▶ Start Webcam** to begin live detection.")

