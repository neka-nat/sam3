import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

from sam3.model.sam3_video_predictor import Sam3VideoPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track objects in a video using a text prompt and save the overlay video."
        )
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to an input video file (MP4/AVI/MOV/...) or frame folder.",
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Text prompt describing the object(s) to track.",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Destination path for the annotated video (e.g., out.mp4).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. Defaults to the HF sam3 checkpoint.",
    )
    parser.add_argument(
        "--bpe-path",
        type=str,
        default=None,
        help="Optional tokenizer path if you do not want the bundled default.",
    )
    parser.add_argument(
        "--prompt-frame",
        type=int,
        default=0,
        help="Frame index (0-based) where the text prompt is anchored.",
    )
    parser.add_argument(
        "--propagation-direction",
        type=str,
        default="both",
        choices=["forward", "backward", "both"],
        help="Direction to propagate the prompt through the video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Track at most this many frames (useful for quick smoke runs).",
    )
    parser.add_argument(
        "--video-loader",
        type=str,
        default="cv2",
        choices=["cv2", "torchcodec"],
        help="Frame loader to use inside the SAM3 predictor.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Opacity of the colored mask overlay.",
    )
    return parser.parse_args()


def color_for_obj(obj_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(int(obj_id))
    color = rng.integers(low=0, high=255, size=3, endpoint=True, dtype=np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def overlay_tracking(
    frame_bgr: np.ndarray, outputs: Dict[str, np.ndarray], alpha: float
) -> np.ndarray:
    obj_ids = outputs["out_obj_ids"]
    masks = outputs["out_binary_masks"]
    boxes = outputs["out_boxes_xywh"]
    probs = outputs["out_probs"]

    alpha = max(0.0, min(alpha, 1.0))
    for idx, obj_id in enumerate(obj_ids):
        mask = masks[idx].astype(bool)
        if not mask.any():
            continue
        color = color_for_obj(int(obj_id))
        frame_bgr[mask] = (
            frame_bgr[mask].astype(np.float32) * (1.0 - alpha)
            + np.array(color, dtype=np.float32) * alpha
        ).astype(np.uint8)

        x, y, w, h = boxes[idx]
        x0 = int(x * frame_bgr.shape[1])
        y0 = int(y * frame_bgr.shape[0])
        x1 = int((x + w) * frame_bgr.shape[1])
        y1 = int((y + h) * frame_bgr.shape[0])
        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color, 2)

        label = f"id:{int(obj_id)} p:{probs[idx]:.2f}"
        cv2.putText(
            frame_bgr,
            label,
            (x0, max(0, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame_bgr


def collect_tracking_results(
    args: argparse.Namespace,
) -> Dict[int, Dict[str, np.ndarray]]:
    predictor = Sam3VideoPredictor(
        checkpoint_path=args.checkpoint,
        bpe_path=args.bpe_path,
        video_loader_type=args.video_loader,
    )
    session_id = predictor.start_session(resource_path=args.video)["session_id"]
    predictor.add_prompt(
        session_id=session_id, frame_idx=args.prompt_frame, text=args.prompt
    )

    frame_outputs: Dict[int, Dict[str, np.ndarray]] = {}
    for result in predictor.propagate_in_video(
        session_id=session_id,
        propagation_direction=args.propagation_direction,
        start_frame_idx=args.prompt_frame,
        max_frame_num_to_track=args.max_frames,
    ):
        frame_outputs[result["frame_index"]] = result["outputs"]

    predictor.close_session(session_id=session_id)
    predictor.shutdown()
    return frame_outputs


def iter_video_frames(video_path: str) -> Iterable[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video at {video_path}")
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame_idx, frame
        frame_idx += 1
    cap.release()


def probe_video(video_path: str, prompt_frame: int) -> Tuple[float, Tuple[int, int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video at {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if frame_count > 0 and prompt_frame >= frame_count:
        raise ValueError(
            f"prompt frame {prompt_frame} is outside the video length "
            f"({frame_count} frames)."
        )
    return fps, (width, height)


def main() -> None:
    args = parse_args()
    fps, (width, height) = probe_video(args.video, args.prompt_frame)
    outputs_by_frame = collect_tracking_results(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for frame_idx, frame in iter_video_frames(args.video):
        if frame_idx in outputs_by_frame:
            frame = overlay_tracking(frame, outputs_by_frame[frame_idx], args.alpha)
        writer.write(frame)
    writer.release()


if __name__ == "__main__":
    main()
