# Basic OpenCV viewer with sliders for rotation and translation.
# Can be easily customizable to different use cases.
import cv2
import numpy as np
import torch
from gsplat import rasterization
import cv2
import tyro
import os
import numpy as np
from typing import Literal
import pycolmap_scene_manager as pycolmap
from utils import (
    load_checkpoint,
    get_rpy_matrix,
    get_viewmat_from_colmap_image,
    prune_by_gradients,
    torch_to_cv,
)

from segment_anything import SamPredictor, sam_model_registry

if not os.path.exists("./checkpoints/sam_vit_h_4b8939.pth"):
    raise ValueError(
        "Please download sam_vit_h_4b8939.pth from https://github.com/facebookresearch/segment-anything and save it in checkpoints folder"
    )
sam = sam_model_registry["vit_h"](
    checkpoint="./checkpoints/sam_vit_h_4b8939.pth"
).cuda()
predictor = SamPredictor(sam)

device = torch.device("cuda:0")


def main(
    data_dir: str = "./data/chair/",  # colmap path
    checkpoint: str = "./data/chair/checkpoint.pth",  # checkpoint path, can generate from original 3DGS repo
    rasterizer: Literal[
        "inria", "gsplat"
    ] = "inria",  # Original or GSplat for checkpoints
    results_dir: str = "./results/chair",
    data_factor: int = 1,
):
    splats = load_checkpoint(
        checkpoint, data_dir, rasterizer=rasterizer, data_factor=data_factor
    )
    splats = prune_by_gradients(splats)

    means = splats["means"].float().to(device)
    opacities = splats["opacity"].to(device)
    quats = splats["rotation"].to(device)
    scales = splats["scaling"].float()

    opacities = torch.sigmoid(opacities)
    scales = torch.exp(scales)
    colors = torch.cat([splats["features_dc"], splats["features_rest"]], 1)

    K = splats["camera_matrix"].float()
    width = int(K[0, 2] * 2)
    height = int(K[1, 2] * 2)

    viewmat_idx = 0
    colmap_images = list(splats["colmap_project"].images.values())

    cv2.namedWindow("Click and Segment", cv2.WINDOW_NORMAL)

    positive_point_prompts = []
    negative_point_prompts = []
    trigger = False

    mask = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal trigger
        if event == cv2.EVENT_LBUTTONDOWN:
            positive_point_prompts.append((x, y))
            trigger = True
        if event == cv2.EVENT_MBUTTONDOWN:
            negative_point_prompts.append((x, y))
            trigger = True

    cv2.setMouseCallback("Click and Segment", mouse_callback)

    accepted_masks = {}

    while True:
        image = colmap_images[viewmat_idx]
        viewmat = get_viewmat_from_colmap_image(image)
        output, _, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmat[None].to(device),
            K[None].to(device),
            width=width,
            height=height,
            render_mode="RGB+D",
            sh_degree=3,
        )

        output_cv = torch_to_cv(output[0, ..., :3])
        output_cv = np.ascontiguousarray(output_cv)
        predictor.set_image(output_cv[..., ::-1])
        if trigger:

            points_np = np.array(positive_point_prompts + negative_point_prompts)
            labels_np = np.array(
                [1 for _ in positive_point_prompts]
                + [0 for _ in negative_point_prompts]
            )
            masks, scores, _ = predictor.predict(points_np, labels_np)
            mask = masks[np.argmax(scores)]
            trigger = False
        if mask is not None:
            output_cv[mask] = 0.5 * output_cv[mask] + 0.5 * np.array([0, 0, 255])
            for x, y in positive_point_prompts:
                cv2.circle(output_cv, (x, y), 5, (0, 255, 0), -1)
            for x, y in negative_point_prompts:
                cv2.circle(output_cv, (x, y), 5, (0, 0, 255), -1)

        cv2.putText(
            output_cv,
            f"n - next view",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            output_cv,
            f"p - previous view",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            output_cv,
            f"a - accept mask",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            output_cv,
            f"q - go to next stage",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            1,
        )
        cv2.imshow("Click and Segment", output_cv)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
        if key == ord("n"):
            viewmat_idx = (viewmat_idx + 1) % len(colmap_images)
            mask = None
            positive_point_prompts = []
            negative_point_prompts = []
        if key == ord("p"):
            viewmat_idx = (viewmat_idx - 1) % len(colmap_images)
            mask = None
            positive_point_prompts = []
            negative_point_prompts = []
        if key == ord("a"):
            accepted_masks[viewmat_idx] = {
                "mask": mask,
                "positive_point_prompts": positive_point_prompts,
                "negative_point_prompts": negative_point_prompts,
            }

        if key in [ord("p"), ord("n")]:
            if viewmat_idx in accepted_masks:
                mask = accepted_masks[viewmat_idx]["mask"]
                positive_point_prompts = accepted_masks[viewmat_idx][
                    "positive_point_prompts"
                ]
                negative_point_prompts = accepted_masks[viewmat_idx][
                    "negative_point_prompts"
                ]

    if len(accepted_masks) == 0:
        raise ValueError("Please accept some masks before proceeding")
    votes = torch.zeros((means.shape[0], 2)).to(device)
    bins = torch.zeros((means.shape[0], 2)).to(device)
    bins.requires_grad = True
    for idx in accepted_masks:
        image = colmap_images[idx]
        mask = accepted_masks[idx]["mask"]
        viewmat = get_viewmat_from_colmap_image(image)
        output, _, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            bins,
            viewmat[None].to(device),
            K[None].to(device),
            width=width,
            height=height,
            render_mode="RGB+D",
            # sh_degree=3,
        )
        mask = torch.from_numpy(mask).float().to(device)
        mask2 = torch.stack([mask, 1 - mask], dim=-1)
        target = mask2 * output[0, ..., :2]
        target = target.sum()
        target.backward()
        votes = votes + bins.grad
        bins.grad.zero_()

    # Show both extraction and deletion based on the mask weights
    cv2.namedWindow("Extraction, Deletion, 2D Mask", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Background weight", "Extraction, Deletion, 2D Mask", 0, 200, lambda x: None)
    cv2.setTrackbarPos("Background weight", "Extraction, Deletion, 2D Mask", 100)

    while True:
        for image in splats["colmap_project"].images.values():
            viewmat = get_viewmat_from_colmap_image(image)

            background_weight = (
                cv2.getTrackbarPos("Background weight", "Extraction, Deletion, 2D Mask") / 100.0
            )

            mask3d = votes[:, 0] > background_weight * votes[:, 1]
            opacities_extracted = opacities.clone()
            opacities_extracted[~mask3d] = 0.0
            opacities_deleted = opacities.clone()
            opacities_deleted[mask3d] = 0.0
            colors_mask = colors[:, 0].clone()
            colors_mask[mask3d] = 1.0
            colors_mask[~mask3d] = 0.0
            with torch.no_grad():
                output, alphas, meta = rasterization(
                    means,
                    quats,
                    scales,
                    opacities_extracted,
                    colors,
                    viewmat[None].to(device),
                    K[None].to(device),
                    width=width,
                    height=height,
                    render_mode="RGB",
                    sh_degree=3,
                )

                output_cv_extracted = torch_to_cv(output[0])

                output, alphas, meta = rasterization(
                    means,
                    quats,
                    scales,
                    opacities_deleted,
                    colors,
                    viewmat[None].to(device),
                    K[None].to(device),
                    width=width,
                    height=height,
                    render_mode="RGB",
                    sh_degree=3,
                )

                output_cv_deleted = torch_to_cv(output[0])

                output, alphas, meta = rasterization(
                    means,
                    quats,
                    scales,
                    opacities,
                    colors_mask,
                    viewmat[None].to(device),
                    K[None].to(device),
                    width=width,
                    height=height,
                    render_mode="RGB",
                    # sh_degree=3,
                )

                output_cv_mask = torch_to_cv(output[0])

                output_cv = cv2.hconcat(
                    [output_cv_extracted, output_cv_deleted, output_cv_mask]
                )
                cv2.imshow("Extraction, Deletion, 2D Mask", output_cv)
                key = cv2.waitKey(10) & 0xFF
                if key == ord("q"):
                    break
        if key == ord("q"):
            break


if __name__ == "__main__":
    tyro.cli(main)
