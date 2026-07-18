"""
analytics.py
------------
Reads saved session reports (sessions/*.json) and prints a trend
summary -- your "Analytics Dashboard" as a terminal report.

Run standalone:
    python analytics.py

If matplotlib is installed, also saves a trend chart to
sessions/trend.png.
"""

import glob
import json
import os

import config


def load_sessions():
    files = sorted(glob.glob(os.path.join(config.SESSIONS_DIR, "session_*.json")))
    sessions = []
    for f in files:
        with open(f) as fh:
            sessions.append(json.load(fh))
    return sessions


def main():
    sessions = load_sessions()
    if not sessions:
        print("No sessions recorded yet. Run main.py first, then press 'q' to save a session.")
        return

    print(f"\n{'Date':<20}{'Duration(min)':<15}{'Posture%':<10}{'EyeContact%':<13}{'Confidence':<12}")
    for s in sessions:
        print(
            f"{s['timestamp']:<20}"
            f"{s['duration_seconds']/60:<15.1f}"
            f"{s['good_posture_percent']:<10.1f}"
            f"{s['eye_contact_percent']:<13.1f}"
            f"{s['average_confidence_score']:<12.1f}"
        )

    if len(sessions) >= 2:
        first, last = sessions[0], sessions[-1]
        print("\nOverall improvement (first session -> latest session):")
        print(f"  Posture:     {first['good_posture_percent']:.1f}% -> {last['good_posture_percent']:.1f}%")
        print(f"  Eye contact: {first['eye_contact_percent']:.1f}% -> {last['eye_contact_percent']:.1f}%")
        print(f"  Confidence:  {first['average_confidence_score']:.1f} -> {last['average_confidence_score']:.1f}")

    try:
        import matplotlib.pyplot as plt

        dates = [s["timestamp"] for s in sessions]
        posture = [s["good_posture_percent"] for s in sessions]
        eye = [s["eye_contact_percent"] for s in sessions]
        conf = [s["average_confidence_score"] for s in sessions]

        plt.figure(figsize=(10, 5))
        plt.plot(posture, marker="o", label="Posture %")
        plt.plot(eye, marker="o", label="Eye Contact %")
        plt.plot(conf, marker="o", label="Confidence")
        plt.xticks(range(len(dates)), [d.split(" ")[0] for d in dates], rotation=45)
        plt.legend()
        plt.title("Progress Over Sessions")
        plt.tight_layout()
        out_path = os.path.join(config.SESSIONS_DIR, "trend.png")
        plt.savefig(out_path)
        print(f"\nChart saved to {out_path}")
    except ImportError:
        print("\n(Install matplotlib for a trend chart too: pip install matplotlib)")


if __name__ == "__main__":
    main()