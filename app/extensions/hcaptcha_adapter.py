# -*- coding: utf-8 -*-
import asyncio
import itertools
from contextlib import suppress
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
from hcaptcha_challenger.agent.challenger import AgentV, RoboticArm
from hcaptcha_challenger.models import CaptchaResponse, PointCoordinate, SpatialPath
from loguru import logger
from playwright.async_api import TimeoutError as PlaywrightTimeoutError


def _longest_contiguous_run(values: np.ndarray) -> list[int]:
    runs: list[list[int]] = []
    for value in values.tolist():
        value = int(value)
        if not runs or value != runs[-1][-1] + 1:
            runs.append([value])
        else:
            runs[-1].append(value)
    return max(runs, key=len, default=[])


def _detect_task_canvas_origin(challenge_screenshot: Path) -> tuple[int, int] | None:
    image = cv2.imread(str(challenge_screenshot))
    if image is None:
        return None

    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    colored_pixels = (hsv[:, :, 1] > 30) & (hsv[:, :, 2] > 20)

    # The prompt header occupies the top of the challenge. The task canvas is the longest
    # colored run below it, regardless of whether hCaptcha renders the 320px or 330px layout.
    row_counts = colored_pixels.sum(axis=1)
    row_indexes = np.flatnonzero(
        (row_counts > width * 0.35) & (np.arange(height) >= int(height * 0.23))
    )
    task_rows = _longest_contiguous_run(row_indexes)
    if len(task_rows) < height * 0.45:
        return None

    task_mask = colored_pixels[task_rows[0] : task_rows[-1] + 1]
    column_indexes = np.flatnonzero(task_mask.sum(axis=0) > len(task_rows) * 0.05)
    if not len(column_indexes):
        return None

    return int(column_indexes.min()), task_rows[0]


def _entity_centers(captcha_payload: Any, crumb_id: int) -> list[tuple[int, int]]:
    tasklist = getattr(captcha_payload, "tasklist", None) or []
    if crumb_id < 0 or crumb_id >= len(tasklist):
        return []

    centers: list[tuple[int, int]] = []
    for entity in getattr(tasklist[crumb_id], "entities", None) or []:
        coords = getattr(entity, "coords", None) or []
        if len(coords) < 2:
            return []
        centers.append((int(coords[0]), int(coords[1])))
    return centers


def _map_canvas_points_to_page(
    points: list[tuple[float, float]],
    *,
    challenge_screenshot: Path,
    challenge_bbox: dict[str, float] | None,
) -> list[tuple[int, int]]:
    if not points or not challenge_bbox:
        return []

    canvas_origin = _detect_task_canvas_origin(challenge_screenshot)
    image = cv2.imread(str(challenge_screenshot))
    if canvas_origin is None or image is None:
        return []

    image_height, image_width = image.shape[:2]
    scale_x = float(challenge_bbox["width"]) / image_width
    scale_y = float(challenge_bbox["height"]) / image_height
    origin_x, origin_y = canvas_origin
    return [
        (
            int(round(float(challenge_bbox["x"]) + (origin_x + x) * scale_x)),
            int(round(float(challenge_bbox["y"]) + (origin_y + y) * scale_y)),
        )
        for x, y in points
    ]


def _payload_source_points(
    *,
    captcha_payload: Any,
    crumb_id: int,
    challenge_screenshot: Path,
    challenge_bbox: dict[str, float] | None,
) -> list[tuple[int, int]]:
    centers = _entity_centers(captcha_payload, crumb_id)
    return _map_canvas_points_to_page(
        centers, challenge_screenshot=challenge_screenshot, challenge_bbox=challenge_bbox
    )


def _correct_drag_source_points(
    paths: list[Any],
    *,
    captcha_payload: Any,
    crumb_id: int,
    challenge_screenshot: Path,
    challenge_bbox: dict[str, float] | None,
) -> list[Any]:
    centers = _entity_centers(captcha_payload, crumb_id)
    if not paths or len(centers) != len(paths) or not challenge_bbox:
        return paths

    resolved_sources = _payload_source_points(
        captcha_payload=captcha_payload,
        crumb_id=crumb_id,
        challenge_screenshot=challenge_screenshot,
        challenge_bbox=challenge_bbox,
    )
    if len(resolved_sources) != len(paths):
        logger.warning("Could not locate hCaptcha drag canvas; keeping model source coordinates")
        return paths

    path_order = sorted(range(len(paths)), key=lambda index: paths[index].start_point.y)
    source_order = sorted(resolved_sources, key=lambda point: point[1])
    for path_index, source in zip(path_order, source_order):
        path = paths[path_index]
        previous = (path.start_point.x, path.start_point.y)
        path.start_point.x, path.start_point.y = source
        logger.info("Corrected hCaptcha drag source from model={} to payload={}", previous, source)

    return paths


def _extract_outline_targets(
    challenge_screenshot: Path,
) -> list[tuple[np.ndarray, tuple[float, float]]]:
    image = cv2.imread(str(challenge_screenshot))
    canvas_origin = _detect_task_canvas_origin(challenge_screenshot)
    if image is None or canvas_origin is None:
        return []

    origin_x, origin_y = canvas_origin
    task_canvas = image[origin_y:, origin_x:]
    hsv = cv2.cvtColor(task_canvas, cv2.COLOR_BGR2HSV)
    outline_mask = ((hsv[:, :, 1] < 100) & (hsv[:, :, 2] > 140)).astype(np.uint8) * 255
    outline_mask[:, int(task_canvas.shape[1] * 0.68) :] = 0
    outline_mask = cv2.morphologyEx(outline_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(outline_mask)
    targets: list[tuple[np.ndarray, tuple[float, float]]] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if not (400 <= area <= 3000 and width >= 35 and height >= 35):
            continue
        if x >= task_canvas.shape[1] * 0.68 or y >= task_canvas.shape[0] * 0.88:
            continue

        component = (labels == index).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        if not moments["m00"]:
            continue
        center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        targets.append((contour, center))

    return targets


def _decode_entity_contour(content: bytes) -> np.ndarray | None:
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] < 4:
        return None

    mask = (image[:, :, 3] > 20).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _match_outline_contours(
    sources: list[np.ndarray], targets: list[tuple[np.ndarray, tuple[float, float]]]
) -> tuple[list[int], list[float]] | None:
    if not sources or len(targets) < len(sources):
        return None

    scores = [
        [cv2.matchShapes(source, target[0], cv2.CONTOURS_MATCH_I1, 0) for target in targets]
        for source in sources
    ]
    assignments = itertools.permutations(range(len(targets)), len(sources))
    best = min(
        assignments, key=lambda item: sum(scores[i][target] for i, target in enumerate(item))
    )
    assigned_scores = [scores[index][target] for index, target in enumerate(best)]
    if max(assigned_scores) > 0.16:
        return None
    return list(best), assigned_scores


async def _resolve_outline_paths(
    *,
    captcha_payload: Any,
    crumb_id: int,
    challenge_screenshot: Path,
    challenge_bbox: dict[str, float] | None,
) -> list[SpatialPath] | None:
    tasklist = getattr(captcha_payload, "tasklist", None) or []
    if crumb_id < 0 or crumb_id >= len(tasklist):
        return None
    entities = getattr(tasklist[crumb_id], "entities", None) or []
    if len(entities) < 2:
        return None

    question = ""
    with suppress(Exception):
        question = captcha_payload.get_requester_question().lower()
    if "outline" not in question:
        return None

    source_points = _payload_source_points(
        captcha_payload=captcha_payload,
        crumb_id=crumb_id,
        challenge_screenshot=challenge_screenshot,
        challenge_bbox=challenge_bbox,
    )
    targets = _extract_outline_targets(challenge_screenshot)
    if len(source_points) != len(entities) or len(targets) < len(entities):
        return None

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            responses = await asyncio.gather(
                *(client.get(str(entity.entity_uri)) for entity in entities)
            )
        for response in responses:
            response.raise_for_status()
        source_contours = [_decode_entity_contour(response.content) for response in responses]
    except Exception as err:
        logger.warning("Could not load hCaptcha drag entities for outline matching: {!r}", err)
        return None

    if any(contour is None for contour in source_contours):
        return None
    matched = _match_outline_contours(source_contours, targets)
    if matched is None:
        logger.warning("hCaptcha outline topology match was not confident; falling back to LLM")
        return None

    assignment, scores = matched
    target_points = _map_canvas_points_to_page(
        [targets[target][1] for target in assignment],
        challenge_screenshot=challenge_screenshot,
        challenge_bbox=challenge_bbox,
    )
    if len(target_points) != len(source_points):
        return None

    paths = [
        SpatialPath(
            start_point=PointCoordinate(x=source[0], y=source[1]),
            end_point=PointCoordinate(x=target[0], y=target[1]),
        )
        for source, target in zip(source_points, target_points)
    ]
    logger.info(
        "Resolved hCaptcha outline paths by topology | scores={} paths={}",
        [round(score, 4) for score in scores],
        [
            {
                "from": (path.start_point.x, path.start_point.y),
                "to": (path.end_point.x, path.end_point.y),
            }
            for path in paths
        ],
    )
    return paths


def _build_drag_prompt(user_prompt: str, *, source_points: list[tuple[int, int]]) -> str:
    details = (
        f"Authoritative draggable centers from the challenge payload: {source_points}. "
        "Use these exact start_point values and reason only about each end_point."
    )
    if "complete the line" in user_prompt.lower():
        details += (
            " For numbered line completion, locate fixed segments 3 and 5 and the two exposed "
            "ends they present. Translate the entire segment 4 between them. Its end_point is "
            "the center of segment 4 after placement and should lie in the geometric corridor "
            "between segments 3 and 5, usually near the midpoint of their numbered circles. "
            "Reject any candidate at marker 3, marker 5, or a distant unrelated empty area."
        )
    return f"{user_prompt}\n\n{details}"


async def _queue_empty_checkcaptcha_response(agent: Any, response: Any) -> bool:
    if "/checkcaptcha/" not in str(getattr(response, "url", "")):
        return False
    body = await response.body()
    if body and body.strip():
        return False

    agent._captcha_response_queue.put_nowait(
        CaptchaResponse.model_validate({"pass": False, "error": "empty_checkcaptcha_response"})
    )
    logger.warning(
        "hCaptcha check returned an empty response; requesting a fresh challenge | status={}",
        getattr(response, "status", "unknown"),
    )
    return True


def _apply_empty_checkcaptcha_patch() -> None:
    if getattr(AgentV._task_handler, "_epic_empty_response_patch", False):
        return

    original_task_handler = AgentV._task_handler

    async def patched_task_handler(self: AgentV, response: Any):
        with suppress(Exception):
            if await _queue_empty_checkcaptcha_response(self, response):
                return
        return await original_task_handler(self, response)

    patched_task_handler._epic_empty_response_patch = True
    AgentV._task_handler = patched_task_handler


def apply_hcaptcha_drag_patch() -> None:
    _apply_empty_checkcaptcha_patch()
    if getattr(RoboticArm.challenge_image_drag_drop, "_epic_drag_source_patch", False):
        return

    async def patched_challenge_image_drag_drop(self: RoboticArm, job_type: Any):
        frame_challenge = await self.get_challenge_frame_locator()
        crumb_count = await self.check_crumb_count()
        cache_key = self.config.create_cache_key(self.captcha_payload)

        for cid in range(crumb_count):
            await self.page.wait_for_timeout(self.config.WAIT_FOR_CHALLENGE_VIEW_TO_RENDER_MS)
            raw, projection = await self._capture_spatial_mapping(frame_challenge, cache_key, cid)
            challenge_bbox = await frame_challenge.locator(
                "//div[@class='challenge-view']"
            ).bounding_box()
            user_prompt = self._match_user_prompt(job_type)
            paths = await _resolve_outline_paths(
                captcha_payload=self.captcha_payload,
                crumb_id=cid,
                challenge_screenshot=raw,
                challenge_bbox=challenge_bbox,
            )
            if paths is None:
                source_points = _payload_source_points(
                    captcha_payload=self.captcha_payload,
                    crumb_id=cid,
                    challenge_screenshot=raw,
                    challenge_bbox=challenge_bbox,
                )
                response = await self._spatial_path_reasoner(
                    challenge_screenshot=raw,
                    grid_divisions=projection,
                    auxiliary_information=_build_drag_prompt(
                        user_prompt, source_points=source_points
                    ),
                )
                logger.debug(f'[{cid+1}/{crumb_count}]ToolInvokeMessage: {response.log_message}')
                self._spatial_path_reasoner.cache_response(
                    path=cache_key.joinpath(f"{cache_key.name}_{cid}_model_answer.json")
                )
                paths = _correct_drag_source_points(
                    response.paths,
                    captcha_payload=self.captcha_payload,
                    crumb_id=cid,
                    challenge_screenshot=raw,
                    challenge_bbox=challenge_bbox,
                )

            for path in paths:
                await self._perform_drag_drop(path)

            with suppress(PlaywrightTimeoutError):
                submit_btn = frame_challenge.locator("//div[@class='button-submit button']")
                await self.click_by_mouse(submit_btn)

    patched_challenge_image_drag_drop._epic_drag_source_patch = True
    RoboticArm.challenge_image_drag_drop = patched_challenge_image_drag_drop
    logger.info("hCaptcha drag topology and response patches loaded")
