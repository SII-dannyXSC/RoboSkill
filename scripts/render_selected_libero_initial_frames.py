#!/usr/bin/env python3
"""Render reproducible initial agent-view frames for selected LIBERO tasks."""

from pathlib import Path
from typing import List

import cv2
from PIL import Image, ImageDraw, ImageFont

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


TASKS = (
    (47, "put cream cheese box in basket", "container pick-and-place"),
    (35, "open the microwave", "hinged door"),
    (22, "close the bottom drawer", "sliding drawer"),
    (20, "turn on the stove", "switch interaction"),
    (36, "put white bowl on plate", "bowl pick-and-place"),
)


def render_task(task_id: int, output_dir: Path) -> Path:
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(task_id)
    init_states = suite.get_task_init_states(task_id)
    env = OffScreenRenderEnv(
        bddl_file_name=suite.get_task_bddl_file_path(task_id),
        camera_heights=512,
        camera_widths=512,
        camera_names=["agentview", "robot0_eye_in_hand"],
        render_gpu_device_id=-1,
    )
    try:
        env.seed(0)
        env.reset()
        observation = env.set_init_state(init_states[0])
        # LIBERO camera observations are vertically flipped relative to display.
        frame = observation["agentview_image"][::-1, :, ::-1]
        output_path = output_dir / f"libero_90_task_{task_id:02d}_initial.png"
        cv2.imwrite(str(output_path), frame)
    finally:
        env.close()
    print(f"task_id={task_id} instruction={task.language} output={output_path}")
    return output_path


def make_contact_sheet(paths: List[Path], output_dir: Path) -> Path:
    tile_width, tile_height = 512, 574
    sheet = Image.new("RGB", (tile_width * 2, tile_height * 3), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for index, ((task_id, short_name, action_type), path) in enumerate(
        zip(TASKS, paths)
    ):
        x = (index % 2) * tile_width
        y = (index // 2) * tile_height
        image = Image.open(path).convert("RGB")
        sheet.paste(image, (x, y))
        draw.text((x + 10, y + 520), f"task {task_id}: {short_name}", fill="black", font=font)
        draw.text((x + 10, y + 546), action_type, fill="#555555", font=font)
    output_path = output_dir / "libero_90_diverse_easy_tasks_initial_frames.png"
    sheet.save(output_path)
    return output_path


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "artifacts" / "initial_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [render_task(task_id, output_dir) for task_id, _, _ in TASKS]
    print(f"contact_sheet={make_contact_sheet(paths, output_dir)}")


if __name__ == "__main__":
    main()
