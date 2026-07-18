"""
face_engine.py
--------------
Runs MediaPipe FaceMesh ONCE per frame and shares the raw landmarks
with behavior_detector.py (face-touch, visibility) and
eye_head_detector.py (gaze/head position), avoiding duplicate model
inference.
"""

import mediapipe as mp

import config


class FaceEngine:
    def __init__(self):
        self._mp_face = mp.solutions.face_mesh
        self._face_mesh = self._mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=config.FACE_MESH_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACE_MESH_MIN_TRACKING_CONFIDENCE,
        )

    def process(self, frame_rgb):
        """Returns raw landmark list for the first detected face, or None."""
        results = self._face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark

    def close(self):
        self._face_mesh.close()