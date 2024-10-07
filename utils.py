import cv2
import numpy as np


def write_mp4v_mp4(frames: list[np.ndarray], save_path: str, fps: int = 30):
    assert save_path.endswith(".mp4")
    h, w = frames[0].shape[:2]
    isColor = len(frames[0].shape) == 3 and frames[0].shape[2] > 1
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use H.264 codec
    video = cv2.VideoWriter(save_path, fourcc, fps, (w, h), isColor)
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        video.write(frame)
    video.release()
    # Ensure the file is closed
    cv2.destroyAllWindows()


def write_avc1_mp4(frames: list[np.ndarray], save_path: str, fps: int = 30):
    assert save_path.endswith(".mp4")
    h, w = frames[0].shape[:2]
    isColor = len(frames[0].shape) == 3 and frames[0].shape[2] > 1
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Use H.264 codec
    video = cv2.VideoWriter(save_path, fourcc, fps, (w, h), isColor)
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        video.write(frame)
    video.release()
    # Ensure the file is closed
    cv2.destroyAllWindows()
