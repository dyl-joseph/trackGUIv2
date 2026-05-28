

def _compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


class BoTSORTForwardTracker:
    def __init__(self, model_name="yolo26x.pt", device=None):
        import torch
        from ultralytics import YOLO

        if device is None:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "0"
            else:
                device = "cpu"

        self.device = device
        self.model = YOLO(model_name)
        self._matched_track_id = None

    def init(self, frame, user_bbox_xyxy):
        results = self.model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            device=self.device,
            verbose=False,
        )

        if results[0].boxes is None or len(results[0].boxes) == 0:
            return False

        boxes = results[0].boxes
        if boxes.id is None:
            return False

        best_iou = 0.0
        best_track_id = None

        for i in range(len(boxes)):
            det_xyxy = boxes.xyxy[i].cpu().numpy()
            iou = _compute_iou(user_bbox_xyxy, det_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_track_id = int(boxes.id[i].item())

        if best_iou < 0.3:
            return False

        self._matched_track_id = best_track_id
        return True

    def update(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            device=self.device,
            verbose=False,
        )

        if results[0].boxes is None or results[0].boxes.id is None:
            return False, None

        boxes = results[0].boxes
        ids = boxes.id.cpu().numpy()

        for i in range(len(boxes)):
            if int(ids[i]) == self._matched_track_id:
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                return True, xyxy
        return False, None

    def reset(self):
        self.model.predictor = None
        self._matched_track_id = None
