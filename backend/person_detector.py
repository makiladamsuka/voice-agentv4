"""YOLOv8 ONNX person detection via OpenCV DNN (CPU-friendly on Raspberry Pi)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class PersonDetection:
    x: float
    y: float
    w: float
    h: float
    confidence: float

    @property
    def cx(self) -> float:
        return self.x + self.w * 0.5

    def aim_y(self, ratio_from_top: float) -> float:
        return self.y + self.h * ratio_from_top

    @property
    def area(self) -> float:
        return self.w * self.h


class PersonDetector:
    """Run YOLOv8n (COCO) and return the largest person box."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        input_size: int = 640,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Person model not found: {path}")
        self._net = cv2.dnn.readNetFromONNX(str(path))
        self._conf = confidence_threshold
        self._nms = nms_threshold
        self._input_size = input_size

    def detect_largest(self, bgr_frame: np.ndarray) -> PersonDetection | None:
        height, width = bgr_frame.shape[:2]
        length = max(height, width)
        square = np.zeros((length, length, 3), dtype=np.uint8)
        square[0:height, 0:width] = bgr_frame
        scale = length / float(self._input_size)

        blob = cv2.dnn.blobFromImage(
            square,
            scalefactor=1 / 255.0,
            size=(self._input_size, self._input_size),
            swapRB=True,  # input is BGR; YOLO ONNX expects RGB
        )
        self._net.setInput(blob)
        outputs = self._net.forward()

        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes: list[list[float]] = []
        scores: list[float] = []
        class_ids: list[int] = []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            _min_score, max_score, _min_loc, (_x, max_class_index) = cv2.minMaxLoc(
                classes_scores
            )
            if max_score < self._conf or int(max_class_index) != PERSON_CLASS_ID:
                continue
            boxes.append(
                [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
            )
            scores.append(float(max_score))
            class_ids.append(int(max_class_index))

        if not boxes:
            return None

        keep = cv2.dnn.NMSBoxes(boxes, scores, self._conf, self._nms)
        if len(keep) == 0:
            return None

        best: PersonDetection | None = None
        for index in np.array(keep).flatten():
            index = int(index)
            box = boxes[index]
            det = PersonDetection(
                x=box[0] * scale,
                y=box[1] * scale,
                w=box[2] * scale,
                h=box[3] * scale,
                confidence=scores[index],
            )
            if best is None or det.area > best.area:
                best = det
        return best
