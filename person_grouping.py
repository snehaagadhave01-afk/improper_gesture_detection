"""
person_grouping.py
--------------------
MediaPipe Hands detects multiple hands but doesn't know which person
each one belongs to (only "Left"/"Right" hand-type). This adds a
lightweight heuristic: hands whose wrists are close together in the
frame are treated as belonging to the same person, letting the app
track and alert on gestures per-person for 2+ people in frame.

NOTE ON ACCURACY: this is proximity clustering, not true person
re-identification (no face matching, no tracking IDs across frames).
It works well when people are reasonably spaced apart. If two people
stand very close together or their hands overlap, they may get grouped
as one "person" for that frame -- an acceptable tradeoff for a
lightweight, no-extra-model solution.
"""

from typing import List


def assign_person_ids(hand_results: List, frame_w: int, cluster_ratio: float) -> List[int]:
    """
    hand_results: list of HandResult objects (each needs .landmarks_px
    with the wrist at index 0).
    Returns a list of 0-indexed person IDs, same order as hand_results.
    """
    if not hand_results:
        return []

    threshold_px = cluster_ratio * frame_w
    wrists = [hr.landmarks_px[0] for hr in hand_results]
    n = len(wrists)
    person_id = [-1] * n
    next_id = 0

    for i in range(n):
        if person_id[i] != -1:
            continue
        person_id[i] = next_id
        for j in range(i + 1, n):
            if person_id[j] != -1:
                continue
            dist = ((wrists[i][0] - wrists[j][0]) ** 2 + (wrists[i][1] - wrists[j][1]) ** 2) ** 0.5
            if dist < threshold_px:
                person_id[j] = next_id
        next_id += 1

    return person_id