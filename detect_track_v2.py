import os
import cv2
import numpy as np
from ultralytics import YOLOWorld

# ==========================================
# 1. HELPER FUNCTIONS & UTILITIES
# ==========================================
def iou(b1, b2):
    """Calculates Intersection over Union for object tracking."""
    x1, y1, x2, y2 = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
    return inter / union if union > 0 else 0

def smooth_box(old_box, new_box, factor):
    """Smoothes bounding box movements over multiple frames."""
    return tuple(int(old_box[i] * (1 - factor) + new_box[i] * factor) for i in range(4))

def draw_box(frame, box, label, conf):
    """Draws a tight bounding box on weapon coordinates."""
    x1, y1, x2, y2 = box
    
    if "handgun" in label.lower() or "pistol" in label.lower():
        clean_label = "HANDGUN"
        color = (0, 0, 255)   # Red Alert Box
    elif "hammer" in label.lower():
        clean_label = "HAMMER"
        color = (255, 0, 0)   # Blue Alert Box
    elif "stick" in label.lower() or "baton" in label.lower():
        clean_label = "STICK"
        color = (0, 255, 255) # Yellow Alert Box
    else:
        return 

    # Draw localized rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Text Tag setup
    label_text = f"{clean_label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label_text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

# ==========================================
# 2. CONFIGURATION & MODEL INITIALIZATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("Initializing Precision Multi-Threshold Weapon Model with Visual Alarms...")
model = YOLOWorld('yolov8m-worldv2.pt')

# Text description anchors for finding weapons exactly
custom_vocab = [
    "handgun pistol", 
    "heavy iron hammer", 
    "wooden stick baton",
    ""  # Negative background class to absorb noise
]
model.set_classes(custom_vocab)

# Setup file streams
source = os.path.join(SCRIPT_DIR, "test.mp4")
output = os.path.join(SCRIPT_DIR, "weapon_detection_tracked.mp4")

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print(f"\n[ERROR] Could not open video at:\n{source}")
    exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Switch codec from 'mp4v' to 'avc1' (H.264) for flawless macOS M-series export integration
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(output, fourcc, fps, (w, h))

# Mac Output verification lock
if not out.isOpened():
    print("\n[RETRY ALERT] AVC1 native engine initial pass blocked. Attempting alternative mp4v layout...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, fps, (w, h))

# Custom hyperparameter settings
hold_frames = 15        
smooth_factor = 0.20    
min_conf = 0.05         

trackers = {}
next_id = 0
frame_count = 0

# ALARM STATE FLAG
alarm_latched = False  # FIXED: Locks true once a threat is found anywhere in the video

# ==========================================
# 3. PROCESSING PIPELINE LOOP
# ==========================================
print("Processing Live Screen Feed (Complete Video Processing Active)...")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    frame_count += 1

    # Run the zero-shot inference framework
    results = model(frame,
                    conf=min_conf,
                    iou=0.15,                 
                    imgsz=640,               
                    device='mps',             # Apple Silicon Core Acceleration
                    augment=False,            
                    agnostic_nms=True,        
                    max_det=5)

    current_detections = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            
            # Gated class thresholds to isolate true objects from environment mistakes
            if label != "":
                if "handgun" in label.lower() or "pistol" in label.lower():
                    if conf < 0.20: continue  
                elif "hammer" in label.lower():
                    if conf < 0.10: continue  
                elif "stick" in label.lower() or "baton" in label.lower():
                    if conf < 0.18: continue
                
                current_detections.append((x1, y1, x2, y2, label, conf))

    # Cross-frame identity sorting logic
    matched_ids = set()
    for (x1, y1, x2, y2, label, conf) in current_detections:
        best_id, best_score = None, 0.20  
        for tid, tdata in trackers.items():
            score = iou((x1, y1, x2, y2), tdata['box'])
            if score > best_score:
                best_score = score
                best_id = tid

        if best_id is not None:
            if trackers[best_id]['label'] != label and trackers[best_id]['conf'] > conf:
                label = trackers[best_id]['label']
                
            smoothed = smooth_box(trackers[best_id]['box'], (x1, y1, x2, y2), smooth_factor)
            trackers[best_id] = {'box': smoothed, 'label': label, 'conf': conf, 'ttl': hold_frames}
            matched_ids.add(best_id)
        else:
            trackers[next_id] = {'box': (x1, y1, x2, y2), 'label': label, 'conf': conf, 'ttl': hold_frames}
            matched_ids.add(next_id)
            next_id += 1

    to_delete = [tid for tid in trackers if tid not in matched_ids]
    for tid in to_delete:
        trackers[tid]['ttl'] -= 1
        if trackers[tid]['ttl'] <= 0:
            del trackers[tid]

    # Draw precisely targeted threat boundaries
    for tid, tdata in trackers.items():
        draw_box(frame, tdata['box'], tdata['label'], tdata['conf'])

    # ==========================================
    # 4. VISUAL DASHBOARD HUD PANEL OVERLAYS
    # ==========================================
    # If a weapon is detected on screen right now, trigger the permanent latch
    if len(trackers) > 0:
        alarm_latched = True

    # Render top banner check based on lock state
    if alarm_latched:
        # Threat Active State: Outer red borders flash
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 8)
        
        banner_text = "ALARM: THREAT DETECTED"
        (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        bg_x1, bg_y1 = (w - tw) // 2 - 20, 15
        bg_x2, bg_y2 = (w + tw) // 2 + 20, 15 + th + 20
        
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 255), -1)
        
        # Fast text blinking cycle every 5 frames
        if (frame_count // 5) % 2 == 0:
            cv2.putText(frame, banner_text, (bg_x1 + 20, bg_y2 - 10), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    else:
        # Secure State: Clean green banner overlay
        banner_text = "STATUS: SYSTEM SECURE"
        (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
        bg_x1, bg_y1 = (w - tw) // 2 - 15, 15
        bg_x2, bg_y2 = (w + tw) // 2 + 15, 15 + th + 15
        
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 180, 0), -1)
        cv2.putText(frame, banner_text, (bg_x1 + 15, bg_y2 - 8), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    # ==========================================
    # 5. WATERMARK BRANDING OVERLAYS (BY UMAR-US)
    # ==========================================
    # Draw bottom left watermark container panel
    wm_text_1 = "ENGINE: BY UMAR-US"
    wm_text_2 = f"ACTIVE THREAT COUNTER: {len(trackers)}"
    
    # Calculate box spacing dimensions dynamically
    (w1, h1), _ = cv2.getTextSize(wm_text_1, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    (w2, h2), _ = cv2.getTextSize(wm_text_2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    
    max_w = max(w1, w2)
    box_y_start = h - h1 - h2 - 35
    
    # Render transparent backing bar on lower window segment
    cv2.rectangle(frame, (15, box_y_start), (max_w + 35, h - 15), (40, 40, 40), -1)
    # Accent strip indicator border piece
    accent_color = (0, 0, 255) if alarm_latched else (0, 180, 0)
    cv2.rectangle(frame, (15, box_y_start), (20, h - 15), accent_color, -1)
    
    # Print the signature branding tags cleanly onto screen pixel buffer
    cv2.putText(frame, wm_text_1, (30, box_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, wm_text_2, (30, box_y_start + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Output presentation window live loop control
    cv2.imshow("Smart Eye - Tight Weapon Localization", frame)
    
    # Write processed frames to file stream
    out.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\nDone! System successfully saved the video file at: {output}")