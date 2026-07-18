"""
scoring.py
----------
Turns the raw issue lists from each detector into 0-100 sub-scores, an
overall confidence score, and short coaching suggestions -- the "Smart
AI Suggestions" layer.
"""

import random
from typing import Dict, List, Tuple

SUGGESTIONS = {
    "slouching": ["Straighten your back", "Sit upright", "Engage your core and sit tall"],
    "bent_neck": ["Lift your head up", "Raise your monitor slightly", "Relax your neck muscles"],
    "uneven_shoulders": ["Keep both shoulders level", "Relax your shoulders"],
    "leaning": ["Center yourself - avoid leaning to one side"],
    "looking_away": ["Maintain eye contact", "Look at the camera"],
    "looking_down": ["Lift your gaze to the camera"],
    "looking_up": ["Bring your gaze back to the camera"],
    "head_tilted": ["Keep your head level"],
    "arms_crossed": ["Try an open posture - uncross your arms", "Stop crossing your arms"],
    "hands_in_pockets": ["Keep your hands visible and relaxed"],
    "fidgeting": ["Relax your hands - avoid restless movement"],
    "poor_openness": ["Open up your posture - relax your arms at your sides"],
    "face_touch": ["Avoid touching your face"],
}

ISSUE_PENALTY = {
    "slouching": 20,
    "bent_neck": 15,
    "uneven_shoulders": 10,
    "leaning": 12,
    "arms_crossed": 12,
    "hands_in_pockets": 8,
    "fidgeting": 15,
    "poor_openness": 8,
    "face_touch": 15,
}


def posture_score(posture_issues: List[Tuple[str, str]]) -> int:
    score = 100
    for name, _ in posture_issues:
        score -= ISSUE_PENALTY.get(name, 10)
    return max(0, score)


def eye_contact_score(eye_contact_ok: bool, current_score: int) -> int:
    """Smoothed toward 100 (good) or 0 (bad) each frame -- responsive but
    not jumpy on-screen number."""
    target = 100 if eye_contact_ok else 0
    return int(current_score + (target - current_score) * 0.05)


def professionalism_score(body_issues: List[Tuple[str, str]], face_touch_active: bool) -> int:
    score = 100
    for name, _ in body_issues:
        score -= ISSUE_PENALTY.get(name, 10)
    if face_touch_active:
        score -= ISSUE_PENALTY["face_touch"]
    return max(0, score)


def confidence_score(posture: int, eye_contact: int, professionalism: int, weights: Dict[str, float]) -> int:
    total = (
        posture * weights["posture"]
        + eye_contact * weights["eye_contact"]
        + professionalism * weights["professionalism"]
    )
    return int(round(total))


def generate_suggestions(
    posture_issues: List[Tuple[str, str]],
    head_issues: List[str],
    body_issues: List[Tuple[str, str]],
    face_touch_active: bool,
    max_suggestions: int = 2,
) -> List[str]:
    active_names = [n for n, _ in posture_issues] + list(head_issues) + [n for n, _ in body_issues]
    if face_touch_active:
        active_names.append("face_touch")

    if not active_names:
        return ["Great posture and presence - keep it up!"]

    suggestions = []
    for name in active_names:
        phrases = SUGGESTIONS.get(name)
        if phrases:
            suggestions.append(random.choice(phrases))
        if len(suggestions) >= max_suggestions:
            break
    return suggestions