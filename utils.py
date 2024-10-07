import cv2
import numpy as np


def read_media(path: str, bgr2rgb: bool = True) -> list[np.ndarray]:
    if path.endswith(".png"):
        frames = [read_png(path)]
    elif path.endswith(".mp4"):
        frames = read_mp4(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    if bgr2rgb:
        frames = [frame[..., ::-1] for frame in frames]
    return frames


def read_mp4(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def read_png(path: str) -> np.ndarray:
    return cv2.imread(path)


def write_media(frames: list[np.ndarray], save_path: str, rgb2bgr: bool = True):
    if rgb2bgr:
        frames = [frame[..., ::-1] for frame in frames]
    if len(frames) == 1:
        if not save_path.endswith(".png"):
            save_path = ".".join(save_path.split(".")[:-1] + ["png"])
        write_png(frames[0], save_path)
    else:
        if not save_path.endswith(".mp4"):
            save_path = ".".join(save_path.split(".")[:-1] + ["mp4"])
        write_avc1_mp4(frames, save_path)


def write_png(img: np.ndarray, save_path: str):
    cv2.imwrite(save_path, img)


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
