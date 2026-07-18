"""
main.py
-------
All-in-one real-time posture / body-language / eye-contact coach.

Run:
    python main.py                # interview mode (default)
    python main.py presentation
    python main.py study
    python main.py office

Keys:
    q -> quit (saves a session report to sessions/)
    c -> recalibrate posture + head baseline
"""

import os
import time
from collections import defaultdict

import cv2
import mediapipe as mp

import config
import scoring
from pose_engine import PoseEngine
from face_engine import FaceEngine
from posture_detector import PostureDetector
from body_language_detector import BodyLanguageDetector
from eye_head_detector import EyeHeadDetector
from behavior_detector import extract_face_points, hand_near_face, MovementMonitor
from gesture_detector import GestureDetector
from session_report import SessionTracker, print_report

FONT = getattr(cv2, config.FONT)

POSTURE_ALERT_COLOR = (0, 140, 255)
HEAD_ALERT_COLOR = (255, 120, 0)
BODY_ALERT_COLOR = (180, 0, 255)
FACE_TOUCH_COLOR = (0, 100, 255)
MOVEMENT_COLOR = (0, 165, 255)
VISIBILITY_COLOR = (0, 0, 200)
BREAK_COLOR = (255, 200, 0)


def get_gesture_for_hand(hand_result, ml_classifier):
    if ml_classifier is not None:
        return ml_classifier.predict(hand_result.landmarks_norm)
    return hand_result.gesture


def draw_hand(frame, hand_result, gesture):
    mp_hands = mp.solutions.hands
    points = hand_result.landmarks_px
    for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], config.CONNECTION_COLOR, 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, config.LANDMARK_COLOR, -1)
    wrist_x, wrist_y = points[0]
    if gesture:
        color = config.IMPROPER_COLOR if gesture["improper"] else config.NORMAL_COLOR
        text = f"{hand_result.handedness}: {gesture['label']}"
    else:
        color = config.NORMAL_COLOR
        text = f"{hand_result.handedness}: Unknown gesture"
    cv2.putText(frame, text, (wrist_x - 40, wrist_y + 40), FONT, 0.6, color, 2)


def draw_posture_skeleton(frame, reading, color):
    cv2.line(frame, reading.nose_px, reading.shoulder_mid_px, color, 3)
    cv2.line(frame, reading.shoulder_mid_px, reading.hip_mid_px, color, 3)
    cv2.line(frame, reading.left_shoulder_px, reading.right_shoulder_px, color, 3)
    for p in (reading.nose_px, reading.shoulder_mid_px, reading.hip_mid_px):
        cv2.circle(frame, p, 6, color, -1)


def draw_alert_panel(frame, alerts):
    h, w = frame.shape[:2]
    if not alerts:
        cv2.rectangle(frame, (0, 0), (w, config.BANNER_HEIGHT), (40, 40, 40), -1)
        cv2.putText(
            frame, "Status: All good - Monitoring...",
            (20, int(config.BANNER_HEIGHT * 0.65)),
            FONT, 0.7, (200, 200, 200), 2,
        )
        return

    for i, (message, color) in enumerate(alerts[:4]):
        y1 = i * config.BANNER_HEIGHT
        y2 = y1 + config.BANNER_HEIGHT
        cv2.rectangle(frame, (0, y1), (w, y2), color, -1)
        cv2.putText(
            frame, f"! {message}",
            (20, y1 + int(config.BANNER_HEIGHT * 0.65)),
            FONT, 0.65, (255, 255, 255), 2,
        )


def draw_big_warning(frame, message, color):
    """Large, bold, centered warning directly on the video feed -- hard to miss."""
    h, w = frame.shape[:2]
    text = message.upper()
    font_scale = 1.1
    thickness = 3

    (text_w, text_h), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    # Shrink font if the message is too wide for the frame
    while text_w > w - 40 and font_scale > 0.5:
        font_scale -= 0.1
        (text_w, text_h), _ = cv2.getTextSize(text, FONT, font_scale, thickness)

    box_y1 = h - 130
    box_y2 = h - 60
    cv2.rectangle(frame, (0, box_y1), (w, box_y2), color, -1)
    cv2.rectangle(frame, (0, box_y1), (w, box_y2), (255, 255, 255), 3)

    text_x = (w - text_w) // 2
    text_y = box_y1 + (box_y2 - box_y1 + text_h) // 2
    cv2.putText(frame, text, (text_x, text_y), FONT, font_scale, (255, 255, 255), thickness)


def draw_scoreboard(frame, scores):
    h, w = frame.shape[:2]
    panel_w, line_h = 260, 28
    x0, y0 = w - panel_w - 10, 10
    cv2.rectangle(frame, (x0, y0), (w - 10, y0 + line_h * 4 + 16), (30, 30, 30), -1)

    def color_for(v):
        if v >= 75:
            return (0, 200, 0)
        if v >= 50:
            return (0, 165, 255)
        return (0, 0, 255)

    labels = [
        ("Confidence", scores["confidence"]),
        ("Eye Contact", scores["eye_contact"]),
        ("Posture", scores["posture"]),
        ("Professionalism", scores["professionalism"]),
    ]
    for i, (label, value) in enumerate(labels):
        y = y0 + 24 + i * line_h
        cv2.putText(frame, f"{label}: {value}%", (x0 + 12, y), FONT, 0.55, color_for(value), 2)


def draw_suggestions(frame, suggestions):
    if not suggestions:
        return
    h, w = frame.shape[:2]
    text = "Tip: " + "  |  ".join(suggestions)
    cv2.putText(frame, text, (20, h - 15), FONT, 0.55, (255, 255, 255), 2)


def main():
    if config.SAVE_SNAPSHOTS and not os.path.exists(config.SNAPSHOT_DIR):
        os.makedirs(config.SNAPSHOT_DIR)

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check CAMERA_INDEX in config.py.")
        return

    need_hands = config.ENABLE_HAND_DETECTION or config.ENABLE_FACE_TOUCH_DETECTION
    hand_detector = GestureDetector() if need_hands else None

    pose_engine = PoseEngine() if (config.ENABLE_POSTURE_DETECTION or config.ENABLE_BODY_LANGUAGE) else None
    face_engine = FaceEngine() if (
        config.ENABLE_EYE_HEAD_DETECTION or config.ENABLE_FACE_TOUCH_DETECTION or config.ENABLE_VISIBILITY_CHECK
    ) else None

    posture_detector = PostureDetector() if config.ENABLE_POSTURE_DETECTION else None
    body_detector = BodyLanguageDetector() if config.ENABLE_BODY_LANGUAGE else None
    head_detector = EyeHeadDetector() if config.ENABLE_EYE_HEAD_DETECTION else None
    movement_monitor = MovementMonitor() if config.ENABLE_MOVEMENT_CHECK else None

    ml_classifier = None
    if config.ENABLE_HAND_DETECTION and config.USE_ML_MODEL:
        from ml_classifier import MLGestureClassifier
        try:
            ml_classifier = MLGestureClassifier()
        except FileNotFoundError as e:
            print(f"WARNING: {e}")

    session = SessionTracker()

    confirm_counters = defaultdict(int)
    last_alert_time = defaultdict(lambda: 0.0)

    visibility_missing_count = 0
    last_visibility_alert_time = 0.0
    face_touch_confirm = 0
    last_face_touch_alert_time = 0.0
    last_movement_alert_time = 0.0
    last_break_reminder_slot = 0
    eye_contact_running_score = 100

    print("Sit up straight and look at the camera for ~1.5s to calibrate.")
    print("Press 'c' anytime to recalibrate. Press 'q' to quit and save your report.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("WARNING: Failed to read frame from webcam.")
            break

        if config.FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        now = time.time()

        active_alerts = []  # (message, color)

        # ---------- Shared model inference (once per frame) ----------
        pose_lm = pose_engine.process(frame_rgb) if pose_engine is not None else None
        face_lm = face_engine.process(frame_rgb) if face_engine is not None else None
        hand_results = hand_detector.process(frame_rgb, w, h) if hand_detector is not None else []
        face_result = extract_face_points(face_lm, w, h)

        # ---------- Hand gestures (optional, off by default) ----------
        if config.ENABLE_HAND_DETECTION:
            seen_this_frame = set()
            gesture_alert = False
            for hand_result in hand_results:
                gesture = get_gesture_for_hand(hand_result, ml_classifier)
                draw_hand(frame, hand_result, gesture)
                if gesture and gesture["improper"]:
                    key = f"{hand_result.handedness}_{gesture['name']}"
                    seen_this_frame.add(key)
                    confirm_counters[key] += 1
                    if confirm_counters[key] >= config.CONFIRM_FRAMES:
                        gesture_alert = True
                        if now - last_alert_time[key] >= config.ALERT_COOLDOWN_SECONDS:
                            last_alert_time[key] = now
                            print(f"[ALERT] Improper gesture confirmed: {key}")
            for key in list(confirm_counters.keys()):
                if key not in seen_this_frame:
                    confirm_counters[key] = 0
            if gesture_alert:
                active_alerts.append(("Improper gesture detected!", config.IMPROPER_COLOR))

        # ---------- Readings (posture + head) ----------
        posture_reading = (
            posture_detector.read(pose_lm, pose_engine.PoseLandmark, w, h)
            if (posture_detector is not None and pose_lm is not None) else None
        )
        head_reading = head_detector.read(face_lm, w, h) if head_detector is not None else None

        # ---------- Calibration (posture and head calibrate independently) ----------
        if posture_detector is not None and posture_reading is not None and not posture_detector.is_calibrated():
            posture_detector.add_calibration_sample(posture_reading)
            if posture_detector.calibration_progress() >= config.CALIBRATION_FRAMES:
                posture_detector.finalize_calibration()
                print("[INFO] Posture calibrated.")

        if head_detector is not None and head_reading is not None and not head_detector.is_calibrated():
            head_detector.add_calibration_sample(head_reading)
            if head_detector.calibration_progress() >= config.CALIBRATION_FRAMES:
                head_detector.finalize_calibration()
                print("[INFO] Eye/head calibrated.")

        posture_ready = posture_detector is None or posture_detector.is_calibrated()
        head_ready = head_detector is None or head_detector.is_calibrated()

        if posture_detector is not None and not posture_detector.is_calibrated():
            pct = int(100 * posture_detector.calibration_progress() / config.CALIBRATION_FRAMES)
            active_alerts.append((f"Calibrating posture... sit straight ({pct}%)", (180, 120, 0)))
        if head_detector is not None and not head_detector.is_calibrated():
            pct = int(100 * head_detector.calibration_progress() / config.CALIBRATION_FRAMES)
            active_alerts.append((f"Calibrating eye contact... look at camera ({pct}%)", (180, 120, 0)))

        # ---------- Posture ----------
        posture_issues = []
        if posture_ready and posture_detector is not None and posture_reading is not None:
            posture_issues = posture_detector.classify(posture_reading)
            color = config.POSTURE_BAD_COLOR if posture_issues else config.POSTURE_GOOD_COLOR
            draw_posture_skeleton(frame, posture_reading, color)
            for _, message in posture_issues:
                active_alerts.append((message, POSTURE_ALERT_COLOR))
        elif posture_reading is not None:
            draw_posture_skeleton(frame, posture_reading, config.POSTURE_GOOD_COLOR)

        # ---------- Head / eye contact ----------
        head_issues, eye_contact_ok = [], True
        if head_ready and head_detector is not None and head_reading is not None:
            head_issues, eye_contact_ok = head_detector.classify(head_reading)
            for name in head_issues:
                msg = scoring.SUGGESTIONS.get(name, [name])[0]
                active_alerts.append((msg, HEAD_ALERT_COLOR))
        eye_contact_running_score = scoring.eye_contact_score(eye_contact_ok, eye_contact_running_score)

        # ---------- Body language ----------
        body_issues = []
        if body_detector is not None:
            body_issues, _ = body_detector.analyze(pose_lm, pose_engine.PoseLandmark, w, h)
            for _, message in body_issues:
                active_alerts.append((message, BODY_ALERT_COLOR))

        # ---------- Visibility ----------
        person_visible = (pose_lm is not None) or (face_lm is not None)
        if config.ENABLE_VISIBILITY_CHECK:
            visibility_missing_count = 0 if person_visible else visibility_missing_count + 1
            if visibility_missing_count >= config.VISIBILITY_CONFIRM_FRAMES:
                active_alerts.insert(0, ("You are not visible! Please stay in frame", VISIBILITY_COLOR))
                if now - last_visibility_alert_time >= config.VISIBILITY_COOLDOWN_SECONDS:
                    last_visibility_alert_time = now
                    print("[ALERT] Person not visible in frame.")

        # ---------- Face touch ----------
        face_touch_active = False
        if config.ENABLE_FACE_TOUCH_DETECTION and face_result.visible:
            touching = any(
                hand_near_face(hr.landmarks_px, face_result, config.FACE_TOUCH_DISTANCE_THRESHOLD)
                for hr in hand_results
            )
            face_touch_confirm = face_touch_confirm + 1 if touching else 0
            if face_touch_confirm >= config.FACE_TOUCH_CONFIRM_FRAMES:
                face_touch_active = True
                active_alerts.append(("Please don't touch your face!", FACE_TOUCH_COLOR))
                if now - last_face_touch_alert_time >= config.FACE_TOUCH_COOLDOWN_SECONDS:
                    last_face_touch_alert_time = now
                    print("[ALERT] Hand near mouth/eyes - avoid touching your face.")

        # ---------- General movement ----------
        if config.ENABLE_MOVEMENT_CHECK and movement_monitor is not None:
            ref_point = face_result.mouth_px if face_result.visible else None
            if movement_monitor.update(ref_point):
                active_alerts.append(("Avoid unwanted movements - please stay still", MOVEMENT_COLOR))
                if now - last_movement_alert_time >= config.MOVEMENT_COOLDOWN_SECONDS:
                    last_movement_alert_time = now
                    print("[ALERT] Excessive movement detected.")

        # ---------- Scores ----------
        p_score = scoring.posture_score(posture_issues)
        prof_score = scoring.professionalism_score(body_issues, face_touch_active)
        conf_score = scoring.confidence_score(p_score, eye_contact_running_score, prof_score, config.SCORE_WEIGHTS)
        scores = {
            "posture": p_score,
            "eye_contact": eye_contact_running_score,
            "professionalism": prof_score,
            "confidence": conf_score,
        }
        draw_scoreboard(frame, scores)

        suggestions = scoring.generate_suggestions(posture_issues, head_issues, body_issues, face_touch_active)
        draw_suggestions(frame, suggestions)

        # ---------- Break reminder (continuous sitting time) ----------
        elapsed_min = (now - session.start_time) / 60
        current_slot = int(elapsed_min // config.BREAK_REMINDER_MINUTES)
        if current_slot > last_break_reminder_slot:
            last_break_reminder_slot = current_slot
            active_alerts.append((f"You've been at this {int(elapsed_min)} min - take a short break!", BREAK_COLOR))
            print("[INFO] Break reminder triggered.")

        draw_alert_panel(frame, active_alerts)
        if active_alerts:
            top_message, top_color = active_alerts[0]
            draw_big_warning(frame, top_message, top_color)

        # ---------- Session tracking ----------
        session.update(posture_ok=not posture_issues, eye_contact_ok=eye_contact_ok, confidence_score=conf_score)

        cv2.imshow("AI Posture & Presence Coach", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            if posture_detector:
                posture_detector.reset_calibration()
            if head_detector:
                head_detector.reset_calibration()

    cap.release()
    if hand_detector is not None:
        hand_detector.close()
    if pose_engine is not None:
        pose_engine.close()
    if face_engine is not None:
        face_engine.close()
    cv2.destroyAllWindows()

    report = session.save_and_compare()
    print_report(report)


if __name__ == "__main__":
    main()