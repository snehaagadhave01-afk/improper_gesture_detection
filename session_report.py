"""
session_report.py
------------------
Tracks stats across a session and writes a JSON report on exit, with a
simple comparison to your previous session.
"""

import json
import os
import time
from statistics import mean

import config


class SessionTracker:
    def __init__(self):
        self.start_time = time.time()
        self.total_frames = 0
        self.good_posture_frames = 0
        self.eye_contact_frames = 0
        self.confidence_samples = []
        self.posture_correction_count = 0
        self._was_bad_posture = False

    def update(self, posture_ok: bool, eye_contact_ok: bool, confidence_score: float):
        self.total_frames += 1

        if posture_ok:
            self.good_posture_frames += 1
            if self._was_bad_posture:
                self.posture_correction_count += 1
            self._was_bad_posture = False
        else:
            self._was_bad_posture = True

        if eye_contact_ok:
            self.eye_contact_frames += 1

        if self.total_frames % 15 == 0:  # sample confidence a few times/sec
            self.confidence_samples.append(confidence_score)

    def _percentages(self):
        if self.total_frames == 0:
            return 0.0, 0.0
        posture_pct = 100 * self.good_posture_frames / self.total_frames
        eye_pct = 100 * self.eye_contact_frames / self.total_frames
        return posture_pct, eye_pct

    def finalize(self) -> dict:
        duration_sec = time.time() - self.start_time
        posture_pct, eye_pct = self._percentages()
        avg_confidence = mean(self.confidence_samples) if self.confidence_samples else 0.0

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(duration_sec, 1),
            "good_posture_percent": round(posture_pct, 1),
            "bad_posture_percent": round(100 - posture_pct, 1),
            "eye_contact_percent": round(eye_pct, 1),
            "average_confidence_score": round(avg_confidence, 1),
            "posture_corrections": self.posture_correction_count,
        }

    def save_and_compare(self) -> dict:
        os.makedirs(config.SESSIONS_DIR, exist_ok=True)
        report = self.finalize()

        previous = self._load_last_session()

        fname = os.path.join(config.SESSIONS_DIR, f"session_{int(time.time())}.json")
        with open(fname, "w") as f:
            json.dump(report, f, indent=2)

        report["previous_comparison"] = self._compare(report, previous)
        return report

    def _load_last_session(self):
        if not os.path.isdir(config.SESSIONS_DIR):
            return None
        files = sorted(
            f for f in os.listdir(config.SESSIONS_DIR)
            if f.startswith("session_") and f.endswith(".json")
        )
        if not files:
            return None
        with open(os.path.join(config.SESSIONS_DIR, files[-1])) as f:
            return json.load(f)

    @staticmethod
    def _compare(current, previous):
        if not previous:
            return "No previous session to compare."
        diff_posture = current["good_posture_percent"] - previous["good_posture_percent"]
        diff_eye = current["eye_contact_percent"] - previous["eye_contact_percent"]
        diff_conf = current["average_confidence_score"] - previous["average_confidence_score"]
        parts = [
            f"Posture {'+' if diff_posture >= 0 else ''}{diff_posture:.1f}%",
            f"Eye contact {'+' if diff_eye >= 0 else ''}{diff_eye:.1f}%",
            f"Confidence {'+' if diff_conf >= 0 else ''}{diff_conf:.1f} pts",
        ]
        return " | ".join(parts)


def print_report(report: dict):
    print("\n===== SESSION REPORT =====")
    print(f"Duration: {report['duration_seconds']/60:.1f} min")
    print(f"Good posture: {report['good_posture_percent']}%")
    print(f"Bad posture: {report['bad_posture_percent']}%")
    print(f"Eye contact: {report['eye_contact_percent']}%")
    print(f"Average confidence score: {report['average_confidence_score']}%")
    print(f"Posture corrections: {report['posture_corrections']}")
    if "previous_comparison" in report:
        print(f"Vs. previous session: {report['previous_comparison']}")
    print("===========================\n")