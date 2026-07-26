"""
preprocessing.py
------------------
Simple low-light enhancement so detection stays more reliable in dim
rooms / uneven lighting -- addresses "detection accuracy decreases
under poor lighting."

Applies CLAHE (adaptive histogram equalization) on the luminance
channel only, so colors aren't distorted, and it's cheap enough to run
every frame in real time.
"""

import cv2

import config


def enhance_frame(frame_bgr):
    if not config.ENABLE_LOW_LIGHT_ENHANCEMENT:
        return frame_bgr

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)