from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2, os, json, time, threading, csv
import numpy as np
import pandas as pd
from ultralytics import YOLO
import pyttsx3
from datetime import datetime
import builtins
import atexit
import torch
from collections import deque
import mediapipe as mp
import torchvision.models as models
import torch.nn as nn
import queue
import mediapipe as mp
import os
import matplotlib.pyplot as plt
from flask import Flask, render_template
import os
from flask import send_from_directory
import matplotlib
matplotlib.use('Agg')  
import cv2
import numpy as np
from ultralytics import YOLO
import joblib
pose_model = YOLO("yolov8s-pose.pt")
classifier = joblib.load("pose_classifiera.pkl")
scaler = joblib.load("scalera.pkl")
print("Classifier + scaler loaded.")
print("Classes:", classifier.classes_)
labels = {
    0: "Standing",
    1: "Turning Around",
    2: "Normal"
}
# -----------------------
# REFERENCE FEATURE EXTRACTOR (EXACTLY LIKE WORKING CODE)
# -----------------------
def extract_features(keypoints):
    keypoints = np.array(keypoints)
    center = np.mean(keypoints, axis=0)
    dist = np.linalg.norm(keypoints - center, axis=1)
    return dist
# -----------------------
# NMS FUNCTION (FROM REFERENCE - SIMPLER & WORKING)
# -----------------------
def iou_xyxy(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - interArea
    if union <= 0:
        return 0.0
    return interArea / union

def nms_xyxy(boxes, scores=None, iou_thresh=0.45):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=float)
    if scores is None:
        scores = np.arange(len(boxes))[::-1].astype(float)
    else:
        scores = np.array(scores, dtype=float)
    idxs = np.argsort(scores)[::-1].tolist()
    keep = []
    while idxs:
        current = idxs.pop(0)
        keep.append(current)
        remove_list = []
        for other in idxs:
            if iou_xyxy(boxes[current], boxes[other]) > iou_thresh:
                remove_list.append(other)
        idxs = [i for i in idxs if i not in remove_list]
    return keep

# -----------------------
# COCO SKELETON (FROM REFERENCE)
# -----------------------
SKELETON = [
    (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 12),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 1), (1, 3),
    (0, 2), (2, 4)
]




NUM_JOINTS = 33
IN_CHANNELS = 3
NUM_CLASSES = 4
SEQ_LEN = 30
CLASS_NAMES = ['standing', 'turningaround', 'normal']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


keypoint_buffers = {}



# Prevent hard exit while debugging
builtins.exit = lambda *args, **kwargs: None

# ========== CONFIG ==========
VIDEO_SOURCES = {
    "front":  "a.mp4",
    "front2": "a.mp4",
    "front3": "a.mp4",   
    "left":  "left.mp4",
    "right": "right.mp4",
}


FRONT_ROWS, FRONT_COLS = 3, 4
SIDE_ROWS,  SIDE_COLS  = 3, 2
WARP_SIZE = 800
STUDENTS_CSV = "students.csv"
CALIB_FILE   = "calib.json"
LOGS_DIR     = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
YOLO_WEIGHTS = "yolov8n.pt"
CONF_THR = 0.35
IOU_THR  = 0.45

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# ----- Load students (ALWAYS use real names for logging) -----
STUDENTS_CSV = "students.csv"

if os.path.exists(STUDENTS_CSV) and os.path.getsize(STUDENTS_CSV) > 0:
    df = pd.read_csv(STUDENTS_CSV)
    students = df.iloc[:, 0].astype(str).str.strip().tolist()
else:
    students = []


# Activity state (for dashboard + de-dup logs)
activity_log = {name: [] for name in students}
activity_lock = threading.Lock()
last_front_status = {}  # {student_name: "present"/"absent"/"moved_out"}

# Load calibration
calib = {"front": [], "left": [], "right": []}
if os.path.exists(CALIB_FILE):
    try:
        with open(CALIB_FILE, "r") as f:
            calib = json.load(f)
    except Exception:
        pass

# OpenCV caps
caps = {}

# Models
yolo = YOLO(YOLO_WEIGHTS)
app = Flask(__name__)

LOG_DIR = "logs"
STATIC_DIR = "static"
CHART_PATH = os.path.join(STATIC_DIR, "attendance_bar.png")
VIOLATION_CHART_PATH = os.path.join(STATIC_DIR, "violation_pie.png")

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)





def read_log_files():
    """Read all CSV log files and combine them into a DataFrame."""
    csv_files = [f for f in os.listdir(LOG_DIR) if f.endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError("Error: No log files found in logs folder.")

    df_list = []
    for file in csv_files:
        file_path = os.path.join(LOG_DIR, file)
        try:
            df = pd.read_csv(file_path)
            if not {"student", "activity"}.issubset(df.columns):
                raise ValueError("CSV must have 'student' and 'activity' columns")
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not df_list:
        raise ValueError("No valid CSV files found.")
    return pd.concat(df_list, ignore_index=True)



import matplotlib.pyplot as plt
from collections import Counter
import os

LOGFILE = "logs/alerts_log.csv"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter
from matplotlib import gridspec

LOGFILE = "logs/alerts_log.csv"   # adjust path if needed
OUT_LINE = "static/activity_line.png"
OUT_DONUT = "static/activity_donut.png"
GROUP_FREQ = "10S"

def generate_dashboard_graphs():
    # read
    df = pd.read_csv(LOGFILE)

    # ---------- normalize & preprocess ----------
    # ensure time column parsed
    df['time'] = pd.to_datetime(df.get('time'), errors='coerce')
    df = df.dropna(subset=['time'])  # drop rows where time couldn't parse

    # normalize activity strings (lowercase, strip)
    df['activity'] = df.get('activity', '').astype(str).str.strip().str.lower()

    # map common variants from your alert log screenshot -> canonical names
    norm_map = {
        "pant,moved out of seat": "moved out of seat",
        "moved out of seat": "moved out of seat",
        "moved out of the seat": "moved out of seat",
        "using phone": "using phone",
        "phone": "using phone",
        "turning around": "turning around",
        "turn around": "turning around",
        "turning": "turning around",
        "standing": "standing",
        "stand": "standing",
        "using chitpaper": "using chitpaper",
        "chitpaper": "using chitpaper",
        # add more mappings here if your CSV uses other phrasings
    }
    df['activity_clean'] = df['activity'].replace(norm_map).where(df['activity'].notna(), other="others")

    # Define desired activity order for plotting and legend (top-to-bottom layering)
    activity_order = ["standing", "turning around", "moved out of seat", "using phone", "using chitpaper"]
    palette = {
        "standing":     "#9b8bff",    # purple
        "turning around":"#3b82f6",   # blue
        "moved out of seat":"#60a5fa",# light-blue
        "using phone":  "#f59e0b",    # amber (visible on dark background)
        "using chitpaper":        "#f87171",    # light red
        "others":       "#6ee7b7"     # mint for any others
    }

    # ---------- DONUT CHART ----------
    activity_counts = Counter(df['activity_clean'])
    # ensure keys in the desired order for donut legend
    donut_labels = [a for a in activity_order if a in activity_counts] + \
                   ([k for k in activity_counts.keys() if k not in activity_order])
    donut_values = [activity_counts[k] for k in donut_labels]

    # safe-guard: if there are no events, create a neutral donut
    if sum(donut_values) == 0:
        donut_labels = ["No Activity"]
        donut_values = [1]

    fig, ax = plt.subplots(figsize=(5,5), dpi=150)
    plt.style.use('dark_background')
    wedges, texts, autotexts = ax.pie(
        donut_values,
        labels=None,                 # we'll put a legend to the side (clean look)
        autopct=lambda pct: f"{pct:.1f}%" if sum(donut_values) > 0 else "",
        startangle=120,
        wedgeprops=dict(width=0.35, edgecolor='w')
    )

    # color wedges to match palette
    for w, lab in zip(wedges, donut_labels):
        w.set_facecolor(palette.get(lab, w.get_facecolor()))

    # center text: total count
    total_count = sum(donut_values)
    ax.text(0, 0, f"Total\n{int(total_count)}", ha='center', va='center', fontsize=10, weight='bold', color='white')
    ax.set_title("Activity Distribution", fontsize=10)

    # legend to the right
    legend_labels = [f"{lab.title()} ({activity_counts.get(lab,0)})" for lab in donut_labels]
    ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(OUT_DONUT, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    # ---------- TIME-SERIES: stacked area (10S buckets) ----------
    # group into regular time buckets
    df.set_index('time', inplace=True)
    grouped = df.groupby([pd.Grouper(freq=GROUP_FREQ), 'activity_clean']).size().unstack(fill_value=0)

    # ensure full time index (no missing buckets)
    start = grouped.index.min()
    end = grouped.index.max()
    if pd.isna(start) or pd.isna(end):
        # no data to plot -> save an empty placeholder and return
        plt.figure(figsize=(10,4), dpi=150)
        plt.style.use('dark_background')
        plt.text(0.5, 0.5, "No activity data", ha='center', va='center', color='white', fontsize=12)
        plt.axis('off')
        plt.savefig(OUT_LINE, bbox_inches='tight')
        plt.close()
        return

    full_index = pd.date_range(start=start, end=end, freq=GROUP_FREQ)
    grouped = grouped.reindex(full_index, fill_value=0)

    # Ensure columns for known activities exist (and an 'others' column)
    for act in activity_order + ["others"]:
        if act not in grouped.columns:
            grouped[act] = 0

    # choose plotting columns in order
    plot_cols = [c for c in activity_order if c in grouped.columns]
    # append others if present
    if 'others' in grouped.columns and grouped['others'].sum() > 0:
        plot_cols.append('others')

    x = grouped.index
    y = grouped[plot_cols].to_numpy()
    cum = np.cumsum(y, axis=1)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14,5), dpi=150)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.07)
    ax = fig.add_subplot(gs[0])

    # layered fill_between from bottom to top
    for i, col in enumerate(plot_cols):
        lower = cum[:, i-1] if i > 0 else np.zeros(len(x))
        upper = cum[:, i]
        ax.fill_between(x, lower, upper,
                        alpha=0.55,
                        linewidth=0.5,
                        label=col.title(),
                        color=palette.get(col, None))

    # x-axis formatting
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Activities")
    ax.set_title("Activities Over Time", fontsize=12, pad=8)
    ax.grid(axis='y', alpha=0.2)

    # legend on the right of the area chart
    

    

    # --- small donut summary on the right (counts by activity) ---
    ax2 = fig.add_subplot(gs[1])
    summary_counts = grouped[plot_cols].sum()
    sizes = summary_counts.values
    labels = [c.title() for c in summary_counts.index]

    # draw donut
    wedges, _ = ax2.pie(sizes, labels=None, startangle=120, wedgeprops=dict(width=0.32, edgecolor='w'))
    for w, colname in zip(wedges, summary_counts.index):
        w.set_facecolor(palette.get(colname, w.get_facecolor()))

    total_count = int(sizes.sum())
    ax2.text(0, 0, f"Total\n{total_count}", ha='center', va='center', fontsize=10, weight='bold', color='white')
    ax2.legend([f"{lab} ({int(v)})" for lab, v in zip(labels, sizes)], loc='center left', bbox_to_anchor=(1.02, 0.5),
               frameon=False, fontsize=8)
    ax2.set_aspect('equal')
    ax2.set_title("Summary", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT_LINE, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

# Example usage
# generate_dashboard_graphs()


def plot_attendance_bar_chart(df):
    """Determine attendance based on majority of 'present'/'absent' logs 
    per student within the first 30 seconds."""

    # Convert timestamp column
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # Calculate seconds from start
    start_time = df["time"].min()
    df["seconds"] = (df["time"] - start_time).dt.total_seconds()

    # Keep only first 30 seconds
    first_30_df = df[df["seconds"] <= 30]

    attendance_status = {}

    # Majority voting per student
    for student, logs in first_30_df.groupby("student"):

        activities = logs["activity"].tolist()

        absent_count = activities.count("absent")
        present_count = activities.count("present")

        # Decide majority
        if absent_count > present_count:
            attendance_status[student] = "absent"
        else:
            attendance_status[student] = "present"

    # Separate student lists
    absent_students = [s for s, st in attendance_status.items() if st == "absent"]
    present_students = [s for s, st in attendance_status.items() if st == "present"]

    # Bar chart values
    counts = pd.Series(attendance_status).value_counts()

    # Plot bar chart
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color=["#8e73de", "#5aacef"])
    plt.title("Attendance Summary (First 30 sec Majority Vote)")
    plt.xlabel("Status")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(CHART_PATH)
    plt.close()

    return absent_students, present_students


def get_students_by_activity(df):
    mapping = {
        "standing": "standing",
        "turning": "turning around",
        "turn around": "turning around",
        "moved out of seat": "moved out of seat",
        "moved out of the seat": "moved out of seat",
        "using phone": "using phone",
        "phone": "using phone",
        "chits": "chits",
        "chit": "chits"
    }

    df["activity"] = df["activity"].astype(str).str.strip().str.lower()
    df["activity_clean"] = df["activity"].replace(mapping)

    result = {}

    for act in ["standing", "turning around", "moved out of seat", "using phone", "chits"]:
        result[act] = df[df["activity_clean"] == act]["student"].unique().tolist()

    return result



# ✅ New function to extract violated and honest students
def analyze_violations(df):
    """Detect violated and honest students and generate pie chart."""
    if "warning_level" not in df.columns:
        return [], [], None

    # Mark violation if 3rd warning or leave hall message present
    violated_students = df[
        df["warning_level"].str.contains("excluded", case=False, na=False)
    ]["student"].unique().tolist()

    all_students = df["student"].unique().tolist()
    honest_students = [s for s in all_students if s not in violated_students]

    # Plot pie chart
    labels = ["Violated", "Honest"]
    sizes = [len(violated_students), len(honest_students)]
    colors = ["#8e73de", "#5aacef"]

    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    plt.title("Violation Summary")
    plt.tight_layout()
    plt.savefig(VIOLATION_CHART_PATH)
    plt.close()

    return violated_students, honest_students, VIOLATION_CHART_PATH







def extract_keypoints(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

# ---------- Helpers ----------
def safe_filename(name: str) -> str:
    base = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    return base.lower() or "unknown"

def save_calib(calib_dict):
    with open(CALIB_FILE, "w") as f:
        json.dump(calib_dict, f)

def load_cap(name):
    if name in caps and caps[name].isOpened():
        return caps[name]
    src = VIDEO_SOURCES.get(name, 0)
    cap = cv2.VideoCapture(src)
    caps[name] = cap
    return cap

def release_caps():
    for c in list(caps.values()):
        try:
            c.release()
        except:
            pass


LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
ALERTS_LOG_FILE = os.path.join(LOGS_DIR, "alerts_log.csv")

alert_tracker = {}  # Tracks alert times and counts
students_blocked_after_third_warning = set()  # Block after 3rd warning
students_blocked_by_absence = set()

def ensure_alert_log_exists():
    if not os.path.exists(ALERTS_LOG_FILE):
        with open(ALERTS_LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "student", "activity", "warning_level"])

def _log_to_individual_csv(student_name, msg, activity):
    fn = os.path.join(LOGS_DIR, f"{safe_filename(student_name)}.csv")
    header = not os.path.exists(fn)
    with open(fn, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header:
            w.writerow(["time", "student", "activity"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), student_name, msg])

# Track activities at same second per student
same_second_activity = {}  # key = (student_name, "YYYY-MM-DD HH:MM:SS"), value = set([activities])

def log_student_activity(student_name, activity):
    if not student_name:
        return

    ensure_alert_log_exists()

    now = datetime.now()
    time_key = now.strftime("%Y-%m-%d %H:%M:%S")  # second-level timestamp

    # Register activity occurrence for same-time filtering
    same_second_activity.setdefault((student_name, time_key), set())
    same_second_activity[(student_name, time_key)].add(activity.lower())

    # ==========================================
    # 0) SAME-SECOND FILTER RULE
    # ==========================================
    current_activities = same_second_activity[(student_name, time_key)]

    # If same second contains both standing + moved out → block moved out alert
    block_moved_out = (
        "standing" in current_activities and 
        "moved out of seat" in activity.lower()
    )

    # ---------------------------
    # 1) ABSENCE-BASED BLOCKING
    # ---------------------------
    personal_log_file = os.path.join(LOGS_DIR, f"{safe_filename(student_name)}.csv")

# Case 1: If the student is already blocked, continue writing logs normally.
    if student_name in students_blocked_by_absence:
        _log_to_individual_csv(student_name, activity, activity)
        return
    
    
    # Case 2: Check if the personal CSV file exists for the student
    if os.path.exists(personal_log_file):
    
        with open(personal_log_file, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
    
        # If only header exists → no need to check
        if len(rows) > 1:
            logs = rows[1:]   # skip header
    
            # Extract first timestamp
            try:
                first_time = datetime.strptime(logs[0][0], "%Y-%m-%d %H:%M:%S")
            except Exception:
                _log_to_individual_csv(student_name, activity, activity)
                return
    
            elapsed = (now - first_time).total_seconds()
    
            # Collect only clean activity strings
            activities_logged = []
            for r in logs:
                if len(r) < 3:
                    continue
                act = r[2].strip().lower()
                if act == "":
                    continue
                activities_logged.append(act)
    
            # ---- NEW LOGIC: Majority ABSENT vs PRESENT ----
            absent_count = activities_logged.count("absent")
            present_count = activities_logged.count("present")
            
            # Count all other messages (excluding absent and present)
            other_count = len(activities_logged) - absent_count - present_count
            
            total_messages = absent_count + present_count + other_count
            
            # Majority rule:
            # If absent > (present + other)
            if absent_count > (present_count + (other_count*0)) and elapsed <= 30:
                students_blocked_by_absence.add(student_name)
                _log_to_individual_csv(student_name, "absent", "absent")
                return
            
    
    

    # ---------------------------
    # 2) THIRD-WARNING BLOCKING
    # ---------------------------
    if student_name in students_blocked_after_third_warning:
        _log_to_individual_csv(student_name, activity, activity)
        return

    # ---------------------------
    # 3) NORMAL ALERT LOGIC
    # ---------------------------
    key = (student_name, activity)
    phone_chit_activities = ["using phone", "using chitpaper"]
    escalate_activities = ["moved out of seat", "standing", "turning around"]

    if key not in alert_tracker:
        alert_tracker[key] = {
            "first_alert_time": None,
            "last_alert_time": None,
            "count": 0
        }

    record = alert_tracker[key]
    first_time = record["first_alert_time"]
    last_time = record["last_alert_time"]
    count = record["count"]

    msg = activity
    warning_level = None

    # Start of alert processing
    if first_time is None:
        record["first_alert_time"] = now
        record["last_alert_time"] = now
        record["count"] = 1
        if activity.lower() in escalate_activities:
            warning_level = "This is your First Warning"
        msg = activity

    else:
        elapsed_since_last = (now - last_time).total_seconds()

        # Immediate third warning for phone/chit
        if activity.lower() in phone_chit_activities:
            if count == 1:
                record["count"] = 1
                record["last_alert_time"] = now
                warning_level = "you are excluded from the writing exam"
                msg = activity
                students_blocked_after_third_warning.add(student_name)
            else:
                _log_to_individual_csv(student_name, activity, activity)
                return

        elif activity.lower() in escalate_activities:
            if elapsed_since_last >= 20:
                count += 1
                record["count"] = count
                record["last_alert_time"] = now

                if count == 2:
                    warning_level = "this is your 2nd warning"
                elif count >= 3:
                    warning_level = "you are excluded from the writing exam"
                    students_blocked_after_third_warning.add(student_name)

                msg = activity

            else:
                _log_to_individual_csv(student_name, activity, activity)
                return

        else:
            warning_level = None
            msg = activity
            record["last_alert_time"] = now
            record["count"] += 1

    # --------------------------------------
    # Always write personal log
    # --------------------------------------
    _log_to_individual_csv(student_name, msg, activity)

    # --------------------------------------
    # SAME-SECOND FILTER APPLIED HERE
    # --------------------------------------
    if block_moved_out:
        return  # ❌ Do NOT write "moved out of seat" alert when same-second standing exists

    # --------------------------------------
    # Write alert to log
    # --------------------------------------
    if warning_level is not None and activity.lower() != "absent" and activity.lower() != "present":
        with open(ALERTS_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([time_key, student_name, activity, warning_level])




speech_queue = queue.Queue()

def speech_thread_func():
    engine = pyttsx3.init('sapi5')
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    while True:
        text = speech_queue.get()
        if text is None:  # signal to exit
            break
        engine.stop()    # stop any ongoing speech immediately
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

speech_thread = threading.Thread(target=speech_thread_func, daemon=True)
speech_thread.start()

def speak_text(text):
    speech_queue.put(text)



def tail_alerts_log_file(log_file_path, poll_interval=1.0):
    last_position = 0
    while True:
        if not os.path.exists(log_file_path):
            time.sleep(poll_interval)
            continue
        with open(log_file_path, "r", newline='', encoding="utf-8") as f:
            f.seek(last_position)
            new_lines = f.readlines()
            last_position = f.tell()

        if new_lines:
            for line in new_lines:
                if line.strip() and not line.lower().startswith("time,student,activity,warning_level"):
                    reader = csv.reader([line])
                    for row in reader:
                        if len(row) >= 4:
                            alert_time, student, activity, warning_level = row[:4]
                            msg = f"Alert! {student} {activity}, {warning_level}"
                            print("DEBUG: Speaking alert:", msg)  # Debug print
                            speak_text(msg)

        time.sleep(poll_interval)


# Start the background thread on app startup
def start_voice_alert_thread():
    t = threading.Thread(target=tail_alerts_log_file, args=(ALERTS_LOG_FILE,), daemon=True)
    t.start()




def compute_homography(quad, size=WARP_SIZE):
    src = np.array(quad, dtype=np.float32)
    dst = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], dtype=np.float32)
    H    = cv2.getPerspectiveTransform(src, dst)
    Hinv = cv2.getPerspectiveTransform(dst, src)
    return H, Hinv

def warp_points(H, pts):
    if len(pts) == 0:
        return np.empty((0,2), dtype=np.float32)
    pts_h = np.hstack([pts, np.ones((len(pts),1), dtype=np.float32)])
    w = (H @ pts_h.T).T
    w = w[:, :2] / w[:, 2:3]
    return w

def backproject_grid_polys(Hinv, size, rows, cols):
    cell_w = size / cols
    cell_h = size / rows
    polys = []
    for i in range(rows):
        for j in range(cols):
            x0, y0 = j*cell_w, i*cell_h
            x1, y1 = (j+1)*cell_w, (i+1)*cell_h
            poly = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], dtype=np.float32)
            poly_back = warp_points(Hinv, poly)
            polys.append(poly_back)
    return polys

def which_cell(warp_pt, size, rows, cols):
    x,y = warp_pt
    if x<0 or y<0 or x>=size or y>=size:
        return None
    cell_w = size / cols
    cell_h = size / rows
    j = int(x // cell_w)
    i = int(y // cell_h)
    if 0<=i<rows and 0<=j<cols:
        return (i,j)
    return None

def draw_polyline(img, pts, color=(0,255,0), thickness=2, closed=False):
    pts_int = pts.astype(np.int32).reshape(-1,1,2)
    cv2.polylines(img, [pts_int], closed, color, thickness, lineType=cv2.LINE_AA)

def put_label(img, text, org, bg=True):
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    if bg:
        (tw, tht), _ = cv2.getTextSize(text, font, scale, th)
        cv2.rectangle(img, (org[0]-4, org[1]-tht-6), (org[0]+tw+4, org[1]+4), (0,0,0), -1)
    cv2.putText(img, text, org, font, scale, (255,255,255), th, cv2.LINE_AA)

def calculate_angle(a, b, c):
    """
    Calculate angle ABC (in degrees)
    a, b, c are (x, y) points
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

# ---------- Core per-camera processing ----------
# app.py  (only showing updated logic parts)

# ---------------- Core per-camera processing ----------------
prior_seat_names = {}
alerts = []
PROCESS_DELAY = 30
start_processing_time = None


def process_frame_for_camera(name, frame, rows=None, cols=None, H=None):
    global start_processing_time   # <-- FIXED (removed stray quote)
   
    alerts = []

    # ----------------------------
    # START TIMER WHEN FIRST FRAME ARRIVES
    # ----------------------------
    if start_processing_time is None:
        start_processing_time = time.time()

    elapsed = time.time() - start_processing_time

    # ----------------------------
    # WAIT FOR DELAY
    # ----------------------------
    if elapsed < PROCESS_DELAY:
        remaining = int(PROCESS_DELAY - elapsed)

        cv2.putText(
            frame,
            f"Processing starts in {remaining}s",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        
        # Stop all processing until delay completes
        return frame, [], [], []

    # ----------------------------
    # PROCESS AFTER DELAY
    # ----------------------------

    
    results = yolo.predict(source=frame, conf=CONF_THR, iou=IOU_THR, classes=[0], verbose=False)
    head_boxes, centers = [], []

    if len(results) and len(results[0].boxes) > 0:
        for b in results[0].boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy()[:4])
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)

            # ---- CENTERED SMALL BOX ----
            small_w = int(w * 0.17)
            small_h = int(h * 0.17)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            x1n = cx - small_w // 2
            x2n = cx + small_w // 2
            y1n = cy - small_h // 2
            y2n = cy + small_h // 2

            # Store head proxy box + center
            head_boxes.append((x1n, y1n, x2n, y2n))
            centers.append((cx, cy))

            # Draw rectangle
            cv2.rectangle(frame, (x1n, y1n), (x2n, y2n), (0, 200, 255), 2)

    if name == "front" and H is not None and rows and cols:
        centers_arr = (
            np.array(centers, dtype=np.float32).reshape(-1, 2)
            if len(centers) > 0
            else np.empty((0, 2), dtype=np.float32)
        )

        warped_centers = (
            warp_points(H, centers_arr)
            if len(centers_arr) > 0
            else np.empty((0, 2), dtype=np.float32)
        )

        cell_w = WARP_SIZE / cols
        cell_h = WARP_SIZE / rows
        seat_centers = {
            (i, j): ((j + 0.5) * cell_w, (i + 0.5) * cell_h)
            for i in range(rows) for j in range(cols)
        }

        seat_to_detections = {(i, j): [] for i in range(rows) for j in range(cols)}

        # Map each warped center to a seat
        for k, wc in enumerate(warped_centers):
            cell = which_cell(wc, WARP_SIZE, rows, cols)
            if cell is not None:
                seat_to_detections[cell].append(k)

        current_seat_names = {}

        # Iterate through seats to detect student status
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx >= len(students):
                    continue

                detections = seat_to_detections[(i, j)]

                if len(detections) == 0:
                    alerts.append(f"{students[idx]} moved out of seat")
                    log_student_activity(students[idx], "moved out of seat")
                    current_seat_names[(i, j)] = None
                else:
                    current_seat_names[(i, j)] = students[idx]

        # Check for moves compared to previous frame
        for seat, name in current_seat_names.items():
            prior_name = prior_seat_names.get(seat)
            if prior_name is not None and name != prior_name:
                alerts.append(f"{prior_name} moved out of seat")
                log_student_activity(prior_name, "moved out of seat")

        prior_seat_names.clear()
        prior_seat_names.update(current_seat_names)

        # Draw bounding boxes & student names
        for (i, j), detection_indices in seat_to_detections.items():
            if detection_indices:
                idx = i * cols + j
                if idx < len(students):
                    student_name = students[idx]
                    for k in detection_indices:
                        if k < len(head_boxes):
                            box = head_boxes[k]
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                            cv2.putText(
                                frame,
                                student_name,
                                (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                            )

    return frame, alerts, head_boxes, centers


# ---------- Stream generators ----------
def generate_setup_frames(cam_name):
    cap = load_cap(cam_name)
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                break
        frame = cv2.resize(frame, (960, 720))
        q = calib.get(cam_name, [])
        if len(q) == 4:
            pts = np.array(q, dtype=np.float32)
            draw_polyline(frame, pts, color=(255,0,0), thickness=3, closed=True)
        ret, buff = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buff.tobytes() + b'\r\n')


activity_start_times = {}  # key: student_name, value: dict of activity start timestamps
ALERT_THRESHOLD = 10 # seconds threshold

def generate_monitor_frames(cam_name):
    cap = load_cap(cam_name)

    # calibration handling
    quad = calib.get("front", []) if cam_name in ["front2", "front3"] else calib.get(cam_name, [])

    if len(quad) == 4:
        H, Hinv = compute_homography(quad, WARP_SIZE)

        if cam_name in ["front", "front2", "front3"]:
            rows, cols = FRONT_ROWS, FRONT_COLS  # 3x4 full grid
            names_for_grid = students[:]

        elif cam_name in ["left"]:
            rows, cols = SIDE_ROWS, SIDE_COLS  # 3x2
            # take col1 & col2
            names_for_grid = [
                students[r * FRONT_COLS + c]
                for r in range(FRONT_ROWS)
                for c in [0, 1]
                if (r * FRONT_COLS + c) < len(students)
            ]

        elif cam_name in ["right"]:
            rows, cols = SIDE_ROWS, SIDE_COLS  # 3x2
            # take col3 & col4
            names_for_grid = [
                students[r * FRONT_COLS + c]
                for r in range(FRONT_ROWS)
                for c in [2, 3]
                if (r * FRONT_COLS + c) < len(students)
            ]

        else:
            rows = cols = None
            names_for_grid = []

        grid_polys_back = backproject_grid_polys(Hinv, WARP_SIZE, rows, cols)

    else:
        H = Hinv = None
        rows = cols = None
        names_for_grid = []

    # ---- main loop ----
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.05)
            continue

        frame = cv2.resize(frame, (960, 720))
        alerts = []

        if H is not None:
            # ---- FRONT CAMERA LOGIC ----
            if cam_name == "front":
                proc_frame, frame_alerts, boxes, centers = process_frame_for_camera(
                    cam_name, frame, rows, cols, H
                )
                alerts.extend(frame_alerts)

                        
            
            
            
                       # -----------------------
            # FRONT CAMERA PROCESSING (COMPLETE SEPARATE BLOCK)
            # -----------------------
            if cam_name == "front2":
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated_frame = frame.copy()
                
                # YOLO Pose Detection (EXACT reference style)
                results_pose = pose_model.predict(
                    source=frame,
                    conf=CONF_THR,
                    iou=IOU_THR,
                    verbose=False,
                    max_det=50
                )
                
                detected_people = []
                centers = []
                
                # Process results EXACTLY like reference code
                if len(results_pose) and len(results_pose[0].boxes) > 0:
                    r = results_pose[0]
                    
                    # Boxes and scores (reference style)
                    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0, 4))
                    try:
                        scores = r.boxes.conf.cpu().numpy()
                    except:
                        scores = np.ones(len(boxes))
                    
                    # Keypoints (XY ONLY - like reference, NO confidence handling)
                    keypoints_all = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else np.empty((0, 17, 2))
                    
                    print(f"DEBUG: {len(boxes)} boxes, {len(keypoints_all)} keypoint sets")
                    
                    # NMS (using reference NMS)
                    keep_indices = nms_xyxy(boxes, scores=scores, iou_thresh=0.45)
                    print(f"DEBUG: NMS kept {len(keep_indices)}")
                    
                    for i in keep_indices:
                        if i >= len(boxes) or i >= len(keypoints_all):
                            continue
                            
                        x1, y1, x2, y2 = map(int, boxes[i])
                        kp = keypoints_all[i]  # (17,2) EXACTLY like reference
                        # COCO keypoint indices (YOLOv8 pose)
                        LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 11, 13, 15
                        RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 12, 14, 16
                        
                        left_knee_angle = calculate_angle(
                            kp[LEFT_HIP],
                            kp[LEFT_KNEE],
                            kp[LEFT_ANKLE]
                        )
                        
                        right_knee_angle = calculate_angle(
                            kp[RIGHT_HIP],
                            kp[RIGHT_KNEE],
                            kp[RIGHT_ANKLE]
                        )
                        
                        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

                        
                        print(f"DEBUG: Person {i} - box ({x1},{y1},{x2},{y2})")
                        
                        # CLASSIFICATION (EXACT REFERENCE LOGIC - NO EXCEPTIONS!)
                        try:
                            features = extract_features(kp)  # Reference feature extractor
                            print(f"DEBUG: Features shape: {features.shape}, range: {features.min():.2f}-{features.max():.2f}")
                            
                            features_scaled = scaler.transform([features])
                            pred = classifier.predict(features_scaled)[0]
                            pred_proba = classifier.predict_proba(features_scaled)[0]
                            
                            print(f"DEBUG: Prediction: {pred}, probs: {pred_proba.round(3)}")
                            
                            # Map to lowercase labels for your system
                            pred_int = int(pred)
                            STRAIGHT_KNEE_THRESHOLD = 170  # degrees

                            if pred_int == 0 and avg_knee_angle >= STRAIGHT_KNEE_THRESHOLD:
                                detected_label = "standing"
                            elif pred_int == 1:
                                detected_label = "turning around"
                            else:
                                detected_label = "normal"

                                
                            print(f"DEBUG: FINAL: {detected_label}")
                            
                        except Exception as e:
                            print(f"ERROR: {e}")
                            detected_label = "normal"
                        
                        detected_people.append((x1, y1, x2, y2, detected_label))
                        
                        # Center for grid mapping
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        centers.append((cx, cy))
                        
                        # Draw bounding box (your colors)
                        color = (0, 255, 0) if detected_label == "normal" else (0, 0, 255)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Single letter label (your style)
                        first_letter = detected_label[0].upper()
                        ((text_w, text_h), _) = cv2.getTextSize(first_letter, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                        text_x = cx - text_w // 2
                        text_y = cy + text_h // 2
                        cv2.putText(annotated_frame, first_letter, (text_x, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
                        
                        # Keypoints (your style)
                        for (x, y) in kp:
                            if not (np.isnan(x) or np.isnan(y)):
                                cv2.circle(annotated_frame, (int(x), int(y)), 2, (255, 200, 200), -1)
                        
                        # Skeleton (reference style)
                        for (p1, p2) in SKELETON:
                            if p1 < 17 and p2 < 17:
                                xA, yA = kp[p1]
                                xB, yB = kp[p2]
                                if not (np.isnan(xA) or np.isnan(yA) or np.isnan(xB) or np.isnan(yB)):
                                    cv2.line(annotated_frame, (int(xA), int(yA)), (int(xB), int(yB)), 
                                            (200, 200, 255), 1)
                
                # ===== YOUR EXISTING WARPING + ACTIVITY TRACKING (UNCHANGED) =====
                if len(centers) > 0 and H is not None and rows is not None and cols is not None:
                    centers_arr = np.array(centers, dtype=np.float32).reshape(-1, 2)
                    warped_centers = warp_points(H, centers_arr)
                else:
                    warped_centers = np.empty((0, 2), dtype=np.float32)
                
                current_time = time.time()
                
                for i, wc in enumerate(warped_centers):
                    seat_cell = which_cell(wc, WARP_SIZE, rows, cols)
                    if seat_cell is None:
                        continue
                
                    seat_index = seat_cell[0] * cols + seat_cell[1]
                    if seat_index >= len(students):
                        continue
                
                    student_name = students[seat_index]
                
                    # Current detected activity
                    if i < len(detected_people):
                        _, _, _, _, detected_label = detected_people[i]
                    else:
                        detected_label = "normal"
                
                    # Skip normal (we track only abnormal: standing/turning)
                    if detected_label == "normal":
                        if student_name in activity_start_times:
                            # Clear all stored timers for this student
                            activity_start_times[student_name] = {}
                        continue
                
                    # Initialize dictionary for student if not present
                    if student_name not in activity_start_times:
                        activity_start_times[student_name] = {}
                
                    # If activity changed → reset timer
                    if detected_label not in activity_start_times[student_name]:
                        # Clear other activities and start new one
                        activity_start_times[student_name] = {
                            detected_label: current_time
                        }
                        continue
                
                    # Activity is same → check duration
                    start_time = activity_start_times[student_name][detected_label]
                    elapsed = current_time - start_time
                
                    if elapsed >= ALERT_THRESHOLD:
                        # Log activity once, then reset to avoid spamming
                        log_student_activity(student_name, detected_label)
                
                        # Reset so the next alert requires another 10 seconds
                        activity_start_times[student_name] = {}
                
                frame = annotated_frame
            
            
            elif cam_name == "front3":
         # presence/absence logic
                results = yolo.predict(
                    source=frame, conf=CONF_THR, iou=IOU_THR, classes=[0], verbose=False
                )
                centers = []
                if len(results) and len(results[0].boxes) > 0:
                    for b in results[0].boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy()[:4])
                        h = max(1, y2 - y1)
                        
                        cx = (x1 + x2) // 2
                        
                        # Stomach point: 55% down from head top
                        cy = y1 + int(h * 0.55)
                        
                        centers.append((cx, cy))
                        
                        # Draw filled rectangle as marker instead of circle
                        rect_width, rect_height = 20, 10  # Size of rectangle
                        top_left = (cx - rect_width // 2, cy - rect_height // 2)
                        bottom_right = (cx + rect_width // 2, cy + rect_height // 2)
                        cv2.rectangle(frame, top_left, bottom_right, (0, 200, 255), -1)
            
                # Warp points for seating grid mapping, if any detected points
                centers_arr = np.array(centers, dtype=np.float32).reshape(-1, 2) if len(centers) > 0 else np.empty((0, 2), dtype=np.float32)
                warped_centers = warp_points(H, centers_arr) if len(centers_arr) > 0 else np.empty((0, 2), dtype=np.float32)
            
                present_cells = set()
                for wc in warped_centers:
                    cell = which_cell(wc, WARP_SIZE, rows, cols)
                    if cell is not None:
                        present_cells.add(cell)
            
                # Draw seat grid + names
                idx = 0
                for i in range(rows):
                    for j in range(cols):
                        poly = grid_polys_back[idx]
                        idx += 1
                        draw_polyline(frame, poly, color=(0, 255, 0), thickness=1, closed=True)
                        tl = poly[0]
                        label_idx = i * cols + j
                        if label_idx < len(students):
                            student_name = students[label_idx]
                            status = "present" if (i, j) in present_cells else "absent"
                            put_label(frame, f"{student_name} [{status}]", (int(tl[0]) + 6, int(tl[1]) + 22))
            
                            if status == "absent":
                                log_student_activity(student_name, "absent")

                            else:
                                log_student_activity(student_name, "present")  
            
            # ---- SIDE CAMERA LOGIC ----
            if cam_name in ["left", "right"]:
                global coco_model, chit_model
            
                try:
                    coco_model
                except NameError:
                    coco_model = YOLO("yolov8n.pt")  # Pretrained COCO (detect phone)
                    chit_model = YOLO("runs/detect/phonepaperchit_model/weights/best.pt")  # Your custom model (ONLY chit class)
            
                detected_objects = []
            
                # ----------------------------------------------------------
                # 1) DETECT PHONE USING COCO MODEL (class name: cell phone)
                # ----------------------------------------------------------
                phone_results = coco_model.predict(frame, conf=0.5)
                
                for r in phone_results:
                    for box in r.boxes:
                        cls_id = int(box.cls)
                        class_name = coco_model.names[cls_id]
            
                        if class_name == "cell phone":
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            detected_objects.append(((x1, y1, x2, y2), "phone"))
            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, "phone", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            
                # ----------------------------------------------------------
                # 2) DETECT CHIT USING YOUR CUSTOM MODEL (ONLY chit class)
                # ----------------------------------------------------------
                chit_results = chit_model.predict(frame, conf=0.5, imgsz=416)
            
                for r in chit_results:
                    for box in r.boxes:
                        cls_id = int(box.cls)
                        label = chit_model.names[cls_id]  # Should be only "chit"
            
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
            
                        detected_objects.append(((x1, y1, x2, y2), label))
            
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            
                # done — return your detected objects


                idx = 0
                for i in range(rows):
                    for j in range(cols):
                        poly = grid_polys_back[idx]; idx += 1
                        draw_polyline(frame, poly, color=(0, 255, 0), thickness=1, closed=True)
                        tl = poly[0]
                        label_idx = i * cols + j
                        if label_idx < len(names_for_grid):
                            student_name = names_for_grid[label_idx]
                            put_label(frame, f"{student_name}", (int(tl[0]) + 6, int(tl[1]) + 22))
            
                            # Check if any object falls inside this grid
                            for (x1, y1, x2, y2), obj_label in detected_objects:
                                box_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                                
                                # Count how much of bbox is inside grid polygon
                                inter = cv2.intersectConvexConvex(poly.astype(np.float32), box_poly)
                                if inter[0] > 0:  # intersection area > 0
                                    box_area = (x2 - x1) * (y2 - y1)
                                    overlap_ratio = inter[0] / box_area
                                    if overlap_ratio > 0.5:  # majority inside seat
                                        log_student_activity(student_name, f"using {obj_label}")
            
            # --- grid overlay for front/front2 only ---
            if cam_name in ["front", "front2"]:
                idx = 0
                for i in range(rows):
                    for j in range(cols):
                        poly = grid_polys_back[idx]; idx += 1
                        draw_polyline(frame, poly, color=(0, 255, 0), thickness=1, closed=True)
                        tl = poly[0]
                        label_idx = i * cols + j
                        label_name = students[label_idx] if label_idx < len(students) else f"({i+1},{j+1})"
                        put_label(frame, f"({i+1},{j+1}) {label_name}", (int(tl[0]) + 6, int(tl[1]) + 22))

        else:
            # fallback if no calibration
            frame, frame_alerts, _, _ = process_frame_for_camera(cam_name, frame)
            alerts.extend(frame_alerts)

        # draw alerts only on main front cam
        if cam_name == "front":
            y = 40
            for a in alerts:
                cv2.putText(frame, str(a), (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                y += 30

        ret, buff = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buff.tobytes() + b'\r\n')



# ---------- Flask routes ----------
@app.route('/')
def index():
    return redirect(url_for('setup_page'))

@app.route('/setup')
def setup_page():
    return render_template('setup.html')  # your HTML file in templates/

# ---------- ROUTE: Save student names ----------
@app.route('/save_students', methods=['POST'])
def save_students():
    global students  # <---- Add this line
    data = request.get_json()
    student_names = data.get('student_names', [])

    if not student_names:
        return jsonify({"message": "No student names received"}), 400

    # Save to CSV safely
    try:
        with open(STUDENTS_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Name"])  # Header
            for name in student_names:
                writer.writerow([name.strip()])

        # ✅ Reload student list immediately
        students = [name.strip() for name in student_names if name.strip()]

        return jsonify({"message": "Student names saved successfully!"})
    except Exception as e:
        return jsonify({"message": f"Error saving students: {e}"}), 500


# ---------- SAFELY READ CSV ANYWHERE ----------
def read_student_csv():
    """Return student list safely without crashing if file empty"""
    if os.path.exists(STUDENTS_CSV) and os.path.getsize(STUDENTS_CSV) > 0:
        try:
            df = pd.read_csv(STUDENTS_CSV)
            if "Name" in df.columns:
                return df["Name"].dropna().astype(str).str.strip().tolist()
            else:
                return df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
        except Exception:
            return []
    return []

# ---------- Example route to test reading students ----------
@app.route('/students')
def list_students():
    students = read_student_csv()
    return jsonify({"students": students})


@app.route('/stream_setup/<cam>')
def stream_setup(cam):
    return Response(generate_setup_frames(cam), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_calib', methods=['POST'])
def save_calib_route():
    global calib
    data = request.json
    if data:
        calib = data
        save_calib(calib)
        return jsonify({"status":"ok"})
    return jsonify({"status":"error"}), 400

@app.route('/monitor')
def monitor_page():
    # Load students from CSV
    if os.path.exists(STUDENTS_CSV) and os.path.getsize(STUDENTS_CSV) > 0:
        df = pd.read_csv(STUDENTS_CSV)
        students = df.iloc[:, 0].astype(str).str.strip().tolist()
    else:
        students = []

    # Load exam details from saved setup.json
    setup_file = "setup.json"
    if os.path.exists(setup_file):
        with open(setup_file, "r") as f:
            setup_data = json.load(f)
    else:
        setup_data = {"exam_name": "Unknown", "subject_code": "N/A", "duration": "N/A"}

    return render_template(
        "monitor.html",
        students=students,
        exam_name=setup_data.get("exam_name", ""),
        subject_code=setup_data.get("subject_code", ""),
        duration=setup_data.get("duration", "")
    )

@app.route('/save_exam_details', methods=['POST'])
def save_exam_details():
    data = request.get_json()
    exam_name = data.get('exam_name', '')
    subject_code = data.get('subject_code', '')
    duration = data.get('duration', '')

    # Save to setup.json
    setup_data = {
        "exam_name": exam_name,
        "subject_code": subject_code,
        "duration": duration
    }

    with open("setup.json", "w") as f:
        json.dump(setup_data, f, indent=2)

    return jsonify({"message": "Exam details saved successfully!"})

@app.route('/stream_monitor/<cam>')
def stream_monitor(cam):
    return Response(generate_monitor_frames(cam), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_monitor/front2')
def stream_monitor_front2():
    return Response(generate_monitor_frames("front2"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_monitor/front3')
def stream_monitor_front3():
    return Response(generate_monitor_frames("front3"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_monitor/left2')
def stream_monitor_left2():
    return Response(generate_monitor_frames("left2"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_monitor/right2')
def stream_monitor_right2():
    return Response(generate_monitor_frames("right2"), mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------- Dashboard ----------------
@app.route("/dashboard")
def dashboard():
    df = read_log_files()

    # Existing charts
    absent_students, present_students = plot_attendance_bar_chart(df)
    violated_students, honest_students, _ = analyze_violations(df)

    # NEW → Generate the two new charts before loading dashboard
    generate_dashboard_graphs()  

    return render_template(
        "dashboard.html",
        absent_students=absent_students,
        present_students=present_students,
        violated_students=violated_students,
        honest_students=honest_students,
    )
@app.route("/filter_activity/<activity>")
def filter_activity(activity):
    df = pd.read_csv(LOGFILE)
    students_map = get_students_by_activity(df)

    activity = activity.lower()
    if activity in students_map:
        return {"students": students_map[activity]}

    return {"students": []}

@app.route('/get_alerts')
def get_alerts():
    return send_from_directory('logs', 'alerts_log.csv')

# cleanup
@atexit.register
def _cleanup():
    release_caps()

# Initialize global student list at startup
students = read_student_csv()
 
if __name__ == "__main__":
    start_voice_alert_thread()
    
    try:
        app.run(debug=True, threaded=True, use_reloader=False)
    finally:
        release_caps()
