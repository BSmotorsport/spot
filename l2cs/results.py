from dataclasses import dataclass

import numpy as np


@dataclass
class GazeResultContainer:
    """Container for gaze inference results produced by the L2CS pipeline."""

    pitch: np.ndarray
    yaw: np.ndarray
    bboxes: np.ndarray
    landmarks: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        # Ensure all arrays share a consistent dtype and dimensionality for downstream ops.
        self.pitch = np.atleast_1d(np.asarray(self.pitch, dtype=np.float32))
        self.yaw = np.atleast_1d(np.asarray(self.yaw, dtype=np.float32))
        self.bboxes = np.asarray(self.bboxes, dtype=np.float32)
        self.landmarks = np.asarray(self.landmarks, dtype=np.float32)
        self.scores = np.asarray(self.scores, dtype=np.float32)

    @property
    def num_faces(self) -> int:
        """Number of detected faces represented in this container."""
        return int(self.pitch.shape[0])
