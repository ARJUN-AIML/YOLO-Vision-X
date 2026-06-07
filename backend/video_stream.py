import threading
import time
from collections import deque
import cv2
import numpy as np
from loguru import logger

class CameraStream:
    def __init__(self, index: int, width: int, height: int, fps: int, max_queue: int) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._fps = fps
        self._queue = deque(maxlen=max_queue)
        self._dropped = 0
        self._stop_event = threading.Event()
        self._thread = None
        self._running = False

    def start(self) -> None:
        if self._running: return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread: self._thread.join()

    def read(self) -> np.ndarray | None:
        if not self._queue: return None
        frame = self._queue.pop()
        self._queue.clear()
        return frame

    def is_running(self) -> bool:
        return self._running

    def _capture_loop(self) -> None:
        import os, glob
        if isinstance(self._index, str) and os.path.isdir(self._index):
            logger.info(f"[CaptureThread] Reading images from directory: {self._index}")
            files = sorted(glob.glob(os.path.join(self._index, "*.jpg")))
            while not self._stop_event.is_set():
                for f in files:
                    if self._stop_event.is_set(): break
                    frame = cv2.imread(f)
                    if frame is not None:
                        frame = cv2.resize(frame, (self._width, self._height))
                        if len(self._queue) == self._queue.maxlen:
                            self._dropped += 1
                        self._queue.append(frame)
                    time.sleep(1/self._fps)
            return

        # Use DSHOW or Any for stability on Windows
        cap = cv2.VideoCapture(self._index, cv2.CAP_DSHOW if isinstance(self._index, int) else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        
        # Flush the buffer to avoid initial black frames
        for _ in range(30):
            cap.grab()

        logger.info("[CaptureThread] Pipeline started.")
        
        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                # Valid frame found
                if len(self._queue) == self._queue.maxlen:
                    self._dropped += 1
                self._queue.append(frame)
            else:
                # Handle camera disconnects gracefully
                logger.warning("Camera frame read failed, retrying...")
                time.sleep(0.5)
            
            time.sleep(1/self._fps)

        cap.release()