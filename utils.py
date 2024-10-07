import pickle
from typing import Sequence

import cv2
import numpy as np


def read_media(path: str) -> Sequence[np.ndarray]:
    if path.endswith(".png"):
        frames = [read_png(path)]
    elif path.endswith(".mp4"):
        frames = read_mp4(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return frames


class MP4Reader:
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        return self

    def __next__(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                self.cap.release()
                raise StopIteration
        else:
            raise StopIteration

    def __len__(self):
        return self.frame_count


def read_mp4(path: str):
    return MP4Reader(path)


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


class AVC1MP4Writer:
    def __init__(self, save_path: str, fps: int = 30, bgr2rgb: bool = True):
        assert save_path.endswith(".mp4")
        self.save_path = save_path
        self.fps = fps
        self.bgr2rgb = bgr2rgb
        self.video = None
        self.initialized = False

    def write(self, frame: np.ndarray):
        if not self.initialized:
            h, w = frame.shape[:2]
            isColor = len(frame.shape) == 3 and frame.shape[2] > 1
            fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Use H.264 codec
            self.video = cv2.VideoWriter(
                self.save_path, fourcc, self.fps, (w, h), isColor
            )
            self.initialized = True

        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        if self.bgr2rgb:
            frame = frame[..., ::-1]
        self.video.write(frame)

    def release(self):
        if self.video is not None:
            self.video.release()
            self.video = None
        cv2.destroyAllWindows()


def write_pickle(data, save_path: str):
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
