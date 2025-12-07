import time

import cv2
import rerun as rr
import torch

from sam3.model_builder import build_sam3_stream_predictor
from sam3.visualization_utils import render_masklet_frame

device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = build_sam3_stream_predictor(device=device)

resp = predictor.handle_request({"type": "start_session"})
session_id = resp["session_id"]

frame_idx = 0

rr.init("sam3_webcam_stream", spawn=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Failed to open video: 0")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

print(f"Video properties: {width}x{height} at {fps} FPS")

processed = 0
peak_memory = 0.0  # GB
start_time = time.time()
try:
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        predictor.handle_request(
            {"type": "add_frame", "session_id": session_id, "frame": frame_rgb}
        )
        if frame_idx == 0:
            predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": "a person",
                }
            )
        resp = predictor.handle_request(
            {
                "type": "run_inference",
                "session_id": session_id,
                "frame_index": frame_idx,
            }
        )
        outputs = resp.get("outputs")
        if outputs is not None:
            overlay = render_masklet_frame(
                frame_rgb, outputs, frame_idx=frame_idx, alpha=0.5
            )
        else:
            overlay = frame_rgb
        rr.set_time_sequence("frame", frame_idx)
        rr.log("webcam/frame", rr.Image(frame_rgb))
        rr.log("webcam/overlay", rr.Image(overlay))
        frame_idx += 1
        processed += 1
        current_peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        peak_memory = max(peak_memory, current_peak_memory)
        print(
            f"Processed frame {frame_idx}. "
            f"Current peak memory: {current_peak_memory:.2f} GB, "
            f"Overall peak memory: {peak_memory:.2f} GB.",
            end="\r",
        )
finally:
    cap.release()
    predictor.handle_request({"type": "close_session", "session_id": session_id})
end_time = time.time()
elapsed = end_time - start_time
fps_est = processed / elapsed if elapsed > 0 else 0.0
print(f"Processed {processed} frames in {elapsed:.2f}s => {fps_est:.2f} FPS")
processed
