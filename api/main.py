from __future__ import annotations

import base64
import math
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

app = FastAPI(title="Shoal Lab local video analysis")
SUPPORTED_EXTENSIONS = {".avi", ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}
SAMPLE_INTERVAL_SECONDS = 0.2
MAX_FRAME_EDGE = 480


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/api/analyze")
async def analyze(
    request: Request,
    filename: str = Query(..., min_length=1, max_length=255),
    frame_rate: float = Query(0, ge=0, le=1000),
    pixel_threshold: int = Query(10, ge=0, le=255),
) -> dict[str, object]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported video extension.")

    descriptor, path_text = tempfile.mkstemp(prefix="shoal-", suffix=suffix)
    path = Path(path_text)
    received_bytes = 0
    try:
        with os.fdopen(descriptor, "wb") as destination:
            async for chunk in request.stream():
                if chunk:
                    destination.write(chunk)
                    received_bytes += len(chunk)

        if received_bytes == 0:
            raise HTTPException(status_code=400, detail="The selected video is empty.")

        try:
            return await run_in_threadpool(
                analyze_video_file,
                path,
                frame_rate,
                pixel_threshold,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        path.unlink(missing_ok=True)


def analyze_video_file(
    path: Path,
    requested_frame_rate: float,
    pixel_threshold: int,
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path), cv2.CAP_ANY)
    if not capture.isOpened():
        capture.release()
        raise ValueError("OpenCV could not decode this video. Check its codec and file integrity.")

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_frame_rate = float(capture.get(cv2.CAP_PROP_FPS))
        effective_frame_rate = requested_frame_rate or source_frame_rate
        if not math.isfinite(effective_frame_rate) or effective_frame_rate <= 0:
            raise ValueError("The video's frame rate could not be read. Enter the source frame rate in Analysis setup.")
        if width <= 0 or height <= 0 or frame_count <= 0:
            raise ValueError("The video does not contain readable frames or duration metadata.")

        frame_step = max(1, round(effective_frame_rate * SAMPLE_INTERVAL_SECONDS))
        points: list[dict[str, float | int | bool | None]] = []
        previous_gray: np.ndarray | None = None
        previous_position: tuple[float, float] | None = None
        previous_track_time: float | None = None
        previous_orientation: float | None = None
        background_model = cv2.createBackgroundSubtractorMOG2(
            history=max(120, min(600, round(effective_frame_rate * 12))),
            varThreshold=pixel_threshold,
            detectShadows=False,
        )
        preview_jpeg = ""
        scale = min(1.0, MAX_FRAME_EDGE / width, MAX_FRAME_EDGE / height)
        sample_width = max(1, round(width * scale))
        sample_height = max(1, round(height * scale))
        sample_area = sample_width * sample_height
        warmup_frames = max(2, round(effective_frame_rate * 0.5))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        minimum_component_area = max(4, round(sample_area * 0.00002))
        maximum_component_area = round(sample_area * 0.18)
        total_distance_pixels = 0.0
        tracked_count = 0
        valid_interval_count = 0
        edge_count = 0
        center_count = 0

        frame_index = 0
        while frame_index < frame_count:
            success, frame = capture.read()
            if not success:
                break

            sampled = cv2.resize(frame, (sample_width, sample_height), interpolation=cv2.INTER_AREA)
            if not preview_jpeg:
                encoded, image = cv2.imencode(".jpg", sampled, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if encoded:
                    preview_jpeg = base64.b64encode(image).decode("ascii")

            learning_rate = 0.08 if frame_index < warmup_frames else 0.001
            foreground = background_model.apply(sampled, learningRate=learning_rate)
            if frame_index < warmup_frames or frame_index % frame_step:
                frame_index += 1
                continue

            foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
            foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = [contour for contour in contours if minimum_component_area <= cv2.contourArea(contour) <= maximum_component_area]
            selected = None
            if candidates:
                if previous_position is None:
                    selected = max(candidates, key=cv2.contourArea)
                else:
                    max_step = math.hypot(sample_width, sample_height) * 0.35
                    reachable = []
                    for contour in candidates:
                        moments = cv2.moments(contour)
                        if moments["m00"] <= 0:
                            continue
                        center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
                        distance = math.dist(center, previous_position)
                        if distance <= max_step:
                            reachable.append((distance, cv2.contourArea(contour), contour, center))
                    if reachable:
                        _, _, selected, selected_center = min(reachable, key=lambda item: item[0] - math.sqrt(item[1]) * 0.15)

            time_seconds = frame_index / effective_frame_rate
            frame_gray = cv2.cvtColor(sampled, cv2.COLOR_BGR2GRAY)
            motion_percent = 0.0
            if previous_gray is not None:
                difference = cv2.absdiff(frame_gray, previous_gray)
                motion_percent = float(round(np.count_nonzero(difference > pixel_threshold) / sample_area * 100, 3))

            x_percent = y_percent = speed_pixels = orientation = turn_rate = area_percent = None
            tracked = selected is not None
            if selected is not None:
                moments = cv2.moments(selected)
                if moments["m00"] > 0:
                    center_x, center_y = selected_center if previous_position is not None else (
                        moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
                    )
                    x_percent = round(center_x / sample_width * 100, 3)
                    y_percent = round(center_y / sample_height * 100, 3)
                    area_percent = round(cv2.contourArea(selected) / sample_area * 100, 4)
                    covariance_angle = 0.5 * math.degrees(math.atan2(
                        2 * moments["mu11"], moments["mu20"] - moments["mu02"],
                    ))
                    orientation = round(covariance_angle, 2)
                    if previous_position is not None:
                        elapsed = max(1 / effective_frame_rate, time_seconds - (previous_track_time or time_seconds))
                        speed_pixels = round(math.dist((center_x, center_y), previous_position) / elapsed, 3)
                        total_distance_pixels += speed_pixels * elapsed
                        valid_interval_count += 1
                        if previous_orientation is not None:
                            angle_delta = (covariance_angle - previous_orientation + 90) % 180 - 90
                            turn_rate = round(angle_delta / elapsed, 3)
                    previous_position = (center_x, center_y)
                    previous_track_time = time_seconds
                    previous_orientation = covariance_angle
                    tracked_count += 1
                    if min(center_x, sample_width - center_x) / sample_width < 0.15 or min(center_y, sample_height - center_y) / sample_height < 0.15:
                        edge_count += 1
                    else:
                        center_count += 1

            points.append({
                "time": round(time_seconds, 3),
                "frame": frame_index,
                "motion": motion_percent,
                "tracked": tracked,
                "x_percent": x_percent,
                "y_percent": y_percent,
                "area_percent": area_percent,
                "orientation_deg": orientation,
                "speed_px_s": speed_pixels,
                "turn_rate_deg_s": turn_rate,
            })
            previous_gray = frame_gray
            frame_index += 1

        if not points:
            raise ValueError("The video was opened, but there were not enough readable frames to analyze.")

        tracked_points = [point for point in points if point["tracked"]]
        speed_values = [float(point["speed_px_s"]) for point in points if point["speed_px_s"] is not None]
        tracked_percent = round(tracked_count / len(points) * 100, 1)
        return {
            "width": width,
            "height": height,
            "frame_rate": effective_frame_rate,
            "duration": round(frame_count / effective_frame_rate, 3),
            "preview_jpeg": preview_jpeg,
            "sample_rate_hz": round(effective_frame_rate / frame_step, 3),
            "tracker": "MOG2 foreground segmentation + centroid association",
            "opencv_version": cv2.__version__,
            "tracked_percent": tracked_percent,
            "tracking_status": "tracked" if tracked_percent >= 60 else "low_coverage" if tracked_percent > 0 else "not_detected",
            "summary": {
                "mean_speed_px_s": round(float(np.mean(speed_values)), 3) if speed_values else None,
                "max_speed_px_s": round(max(speed_values), 3) if speed_values else None,
                "distance_px": round(total_distance_pixels, 2) if speed_values else None,
                "mean_area_percent": round(float(np.mean([float(point["area_percent"]) for point in tracked_points])), 4) if tracked_points else None,
                "edge_occupancy_percent": round(edge_count / tracked_count * 100, 1) if tracked_count else None,
                "center_occupancy_percent": round(center_count / tracked_count * 100, 1) if tracked_count else None,
                "valid_speed_intervals": valid_interval_count,
            },
            "points": points,
        }
    finally:
        capture.release()