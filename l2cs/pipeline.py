from __future__ import annotations

import pathlib
from typing import Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

from face_detection import RetinaFace

from .results import GazeResultContainer
from .utils import getArch, prep_input_numpy


class Pipeline:
    """Inference helper that wraps face detection and L2CS gaze estimation."""

    def __init__(
        self,
        weights: pathlib.Path,
        arch: str,
        device: Union[str, torch.device] = "cpu",
        include_detector: bool = True,
        confidence_threshold: float = 0.5,
    ) -> None:
        # Normalise device input so downstream torch utilities behave correctly.
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.weights = weights
        self.include_detector = include_detector
        self.confidence_threshold = confidence_threshold

        # Prepare the L2CS backbone.
        self.model = getArch(arch, 90)
        state_dict = torch.load(self.weights, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Pre-compute helper tensors used for continuous gaze estimation.
        self.softmax = nn.Softmax(dim=1)
        self.idx_tensor = torch.arange(90, dtype=torch.float32, device=self.device)

        # Optionally bootstrap RetinaFace for face detection.
        if self.include_detector:
            if self.device.type == "cpu":
                self.detector = RetinaFace()
            else:
                gpu_id = 0 if self.device.index is None else self.device.index
                self.detector = RetinaFace(gpu_id=gpu_id)

    def step(self, frame: np.ndarray) -> GazeResultContainer:
        """Run a single inference step on a frame and return structured results."""

        face_imgs = []
        bboxes = []
        landmarks = []
        scores = []

        pitch = np.empty((0,), dtype=np.float32)
        yaw = np.empty((0,), dtype=np.float32)

        if self.include_detector:
            faces = self.detector(frame)

            if faces is not None:
                for box, landmark, score in faces:
                    if score < self.confidence_threshold:
                        continue

                    x_min = max(int(box[0]), 0)
                    y_min = max(int(box[1]), 0)
                    x_max = int(box[2])
                    y_max = int(box[3])

                    # Crop and prepare the detected face for the model.
                    crop = frame[y_min:y_max, x_min:x_max]
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop = cv2.resize(crop, (224, 224))
                    face_imgs.append(crop)

                    bboxes.append(box)
                    landmarks.append(landmark)
                    scores.append(score)

                if face_imgs:
                    pitch, yaw = self.predict_gaze(np.stack(face_imgs))
        else:
            pitch, yaw = self.predict_gaze(frame)

        if bboxes:
            bboxes_arr = np.stack(bboxes).astype(np.float32)
        else:
            bboxes_arr = np.empty((0, 4), dtype=np.float32)

        if landmarks:
            landmarks_arr = np.stack(landmarks).astype(np.float32)
        else:
            landmarks_arr = np.empty((0, 5, 2), dtype=np.float32)

        scores_arr = np.array(scores, dtype=np.float32)

        results = GazeResultContainer(
            pitch=pitch,
            yaw=yaw,
            bboxes=bboxes_arr,
            landmarks=landmarks_arr,
            scores=scores_arr,
        )
        return results

    def predict_gaze(self, frame: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict gaze for a stack of faces following the official L2CS post-processing."""

        if isinstance(frame, np.ndarray):
            img = prep_input_numpy(frame, self.device)
        elif isinstance(frame, torch.Tensor):
            img = frame.to(self.device)
        else:
            raise RuntimeError("Invalid dtype for input")

        with torch.no_grad():
            gaze_pitch, gaze_yaw = self.model(img)
            pitch_prob = self.softmax(gaze_pitch)
            yaw_prob = self.softmax(gaze_yaw)

            pitch_cont = torch.sum(pitch_prob * self.idx_tensor, dim=1) * 4 - 180
            yaw_cont = torch.sum(yaw_prob * self.idx_tensor, dim=1) * 4 - 180

        pitch_rad = pitch_cont.cpu().numpy() * np.pi / 180.0
        yaw_rad = yaw_cont.cpu().numpy() * np.pi / 180.0

        return pitch_rad, yaw_rad
