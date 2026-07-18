"""
utils.py
--------
Small, dependency-free helper functions for working with MediaPipe hand
landmarks: figuring out which fingers are extended, and pattern matching
against the gesture rule table in config.py.
"""

from typing import List, Tuple, Optional


# MediaPipe hand landmark indices we need
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


def get_finger_states(landmarks, handedness_label: str) -> Tuple[bool, bool, bool, bool, bool]:
    """
    Given a list of 21 (x, y) landmark points (normalized 0-1, image
    coordinates) and the handedness ("Left"/"Right" as reported by
    MediaPipe), return a tuple of 5 booleans:
        (thumb, index, middle, ring, pinky)
    True  = finger extended
    False = finger folded
    """
    lm = landmarks  # alias

    # --- Thumb: compare x-position of tip vs the IP joint. Direction
    # depends on which hand it is, because MediaPipe's "Left"/"Right"
    # label is from the CAMERA's point of view (mirrored image).
    if handedness_label == "Right":
        thumb_open = lm[THUMB_TIP][0] < lm[THUMB_IP][0]
    else:
        thumb_open = lm[THUMB_TIP][0] > lm[THUMB_IP][0]

    # --- Other four fingers: tip above (smaller y) than PIP joint means
    # extended. This works reliably as long as the hand is roughly
    # upright, which covers the vast majority of webcam use cases.
    index_open = lm[INDEX_TIP][1] < lm[INDEX_PIP][1]
    middle_open = lm[MIDDLE_TIP][1] < lm[MIDDLE_PIP][1]
    ring_open = lm[RING_TIP][1] < lm[RING_PIP][1]
    pinky_open = lm[PINKY_TIP][1] < lm[PINKY_PIP][1]

    return (thumb_open, index_open, middle_open, ring_open, pinky_open)


def pattern_matches(state: Tuple[bool, bool, bool, bool, bool],
                     pattern: Tuple[Optional[bool], ...]) -> bool:
    """
    Compare a finger-state tuple against a rule pattern that may contain
    None as a wildcard for "don't care".
    """
    for state_val, pattern_val in zip(state, pattern):
        if pattern_val is None:
            continue
        if state_val != pattern_val:
            return False
    return True


def classify_gesture(state: Tuple[bool, bool, bool, bool, bool], rules: List[dict]) -> Optional[dict]:
    """
    Walk the rule table and return the first matching rule dict, or None
    if no rule matches the current finger state.
    """
    for rule in rules:
        if pattern_matches(state, rule["pattern"]):
            return rule
    return None