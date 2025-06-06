# Basic OpenCV viewer with sliders for rotation and translation.
# Can be easily customizable to different use cases.
import torch
from gsplat import rasterization
import cv2
import tyro
import numpy as np
import json
from typing import Literal
import pycolmap_scene_manager as pycolmap
from scipy.spatial.transform import Rotation as scipyR

device = torch.device("cuda:0")

def get_rpy_matrix(roll, pitch, yaw):
    roll_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(roll), -np.sin(roll), 0],
            [0, np.sin(roll), np.cos(roll), 0],
            [0, 0, 0, 1.0],
        ])
    
    pitch_matrix = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch), 0],
            [0, 1, 0, 0],
            [-np.sin(pitch), 0, np.cos(pitch), 0],
            [0, 0, 0, 1.0],
        ])
    yaw_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0, 0],
            [np.sin(yaw), np.cos(yaw), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1.0],
        ]

    )

    return yaw_matrix @ pitch_matrix @ roll_matrix



def _detach_tensors_from_dict(d, inplace=True):
    if not inplace:
        d = d.copy()
    for key in d:
        if isinstance(d[key], torch.Tensor):
            d[key] = d[key].detach()
    return d


def load_checkpoint(checkpoint: str, data_dir: str, rasterizer: Literal["original", "gsplat"]="original", data_factor: int = 1):

    colmap_project = pycolmap.SceneManager(f"{data_dir}/sparse/0")
    colmap_project.load_cameras()
    colmap_project.load_images()
    colmap_project.load_points3D()
    model = torch.load(checkpoint) # Make sure it is generated by 3DGS original repo
    if rasterizer == "original":
        model_params, _ = model
        splats = {
            "active_sh_degree": model_params[0],
            "means": model_params[1],
            "features_dc": model_params[2],
            "features_rest": model_params[3],
            "scaling": model_params[4],
            "rotation": model_params[5],
            "opacity": model_params[6].squeeze(1),
        }
    elif rasterizer == "gsplat":

        model_params = model["splats"]
        splats = {
            "active_sh_degree": 3,
            "means": model_params["means"],
            "features_dc": model_params["sh0"],
            "features_rest": model_params["shN"],
            "scaling": model_params["scales"],
            "rotation": model_params["quats"],
            "opacity": model_params["opacities"],
        }
    else:
        raise ValueError("Invalid rasterizer")

    _detach_tensors_from_dict(splats)

    # Assuming only one camera
    for camera in colmap_project.cameras.values():
        camera_matrix = torch.tensor(
            [
                [camera.fx, 0, camera.cx],
                [0, camera.fy, camera.cy],
                [0, 0, 1],
            ]
        )
        break

    camera_matrix[:2,:3] /= data_factor

    splats["camera_matrix"] = camera_matrix
    splats["colmap_project"] = colmap_project
    splats["colmap_dir"] = data_dir

    return splats

def create_checkerboard(width, height, size=64):
    checkerboard = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, size):
        for x in range(0, width, size):
            if (x // size + y // size) % 2 == 0:
                checkerboard[y:y + size, x:x + size] = 255
            else:
                checkerboard[y:y + size, x:x + size] = 128
    return checkerboard


def main(data_dir: str = "./data/chair", # colmap path
        checkpoint: str = "./data/chair/checkpoint.pth", # checkpoint path, can generate from original 3DGS repo
        rasterizer: Literal["original", "gsplat"] = "original", # Original or GSplat for checkpoints
        mask_path: str = "./results/chair/mask3d.pth",
        apply_mask: bool = True,
        invert: bool = False,
        use_checkerboard_background: bool = True,
        data_factor: int = 1):
    """Program to view the extracted 3D segment.

    Args:
        data_dir: Path to the colmap project.
        checkpoint: checkpoint path, can generate from original 3DGS repo or using gsplat.
        rasterizer: The rasterizer which is used to generate the checkpoint.
        mask_path: Path to the mask file.
        apply_mask: Apply the mask to the splats.
        invert: Invert the mask.
        use_checkerboard_background: Use checkerboard background.
        data_factor: Factor to scale the resolution down.
    """

    torch.set_default_device("cuda")
    torch.set_grad_enabled(False)

    splats = load_checkpoint(checkpoint, data_dir, rasterizer=rasterizer, data_factor=data_factor)

    show_anaglyph = False


    means = splats["means"].float()
    opacities = splats["opacity"]
    quats = splats["rotation"]
    scales = splats["scaling"].float()

    opacities = torch.sigmoid(opacities)
    scales = torch.exp(scales)
    colors = torch.cat([splats["features_dc"], splats["features_rest"]], 1)
    if apply_mask:
        mask = torch.load(mask_path)


        if invert:
            mask = ~mask

        means = means[mask]
        opacities = opacities[mask]
        quats = quats[mask]
        scales = scales[mask]
        colors = colors[mask]

    cv2.namedWindow("Viewer", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Roll", "Viewer", 0, 180, lambda x: None)
    cv2.createTrackbar("Pitch", "Viewer", 0, 180, lambda x: None)
    cv2.createTrackbar("Yaw", "Viewer", 0, 180, lambda x: None)
    cv2.createTrackbar("X", "Viewer", 0, 1000, lambda x: None)
    cv2.createTrackbar("Y", "Viewer", 0, 1000, lambda x: None)
    cv2.createTrackbar("Z", "Viewer", 0, 1000, lambda x: None)
    cv2.createTrackbar("Scaling", "Viewer", 100, 100, lambda x: None)

    cv2.setTrackbarMin("Roll", "Viewer", -180)
    cv2.setTrackbarMax("Roll", "Viewer", 180)
    cv2.setTrackbarMin("Pitch", "Viewer", -180)
    cv2.setTrackbarMax("Pitch", "Viewer", 180)
    cv2.setTrackbarMin("Yaw", "Viewer", -180)
    cv2.setTrackbarMax("Yaw", "Viewer", 180)
    cv2.setTrackbarMin("X", "Viewer", -1000)
    cv2.setTrackbarMax("X", "Viewer", 1000)
    cv2.setTrackbarMin("Y", "Viewer", -1000)
    cv2.setTrackbarMax("Y", "Viewer", 1000)
    cv2.setTrackbarMin("Z", "Viewer", -1000)
    cv2.setTrackbarMax("Z", "Viewer", 1000)


    K = splats["camera_matrix"].float()


    width = int(K[0, 2] * 2)
    height = int(K[1, 2] * 2)

    def update_trackbars_from_viewmat(world_to_camera):
        # if torch tensor is passed, convert to numpy
        if isinstance(world_to_camera, torch.Tensor):
            world_to_camera = world_to_camera.cpu().numpy()
        r = scipyR.from_matrix(world_to_camera[:3,:3])
        roll, pitch, yaw = r.as_euler('xyz')
        cv2.setTrackbarPos("Roll", "Viewer", np.rad2deg(roll).astype(int))
        cv2.setTrackbarPos("Pitch", "Viewer", np.rad2deg(pitch).astype(int))
        cv2.setTrackbarPos("Yaw", "Viewer", np.rad2deg(yaw).astype(int))
        cv2.setTrackbarPos("X", "Viewer", int(world_to_camera[0, 3]*100))
        cv2.setTrackbarPos("Y", "Viewer", int(world_to_camera[1, 3]*100))
        cv2.setTrackbarPos("Z", "Viewer", int(world_to_camera[2, 3]*100))

    while True:
        roll = cv2.getTrackbarPos("Roll", "Viewer")
        pitch = cv2.getTrackbarPos("Pitch", "Viewer")
        yaw = cv2.getTrackbarPos("Yaw", "Viewer")

        roll_rad = np.deg2rad(roll)
        pitch_rad = np.deg2rad(pitch)
        yaw_rad = np.deg2rad(yaw)

        viewmat = (
            torch.tensor(get_rpy_matrix(roll_rad, pitch_rad, yaw_rad))
            .float()
            .to(device)
        )
        viewmat[0, 3] = cv2.getTrackbarPos("X", "Viewer") / 100.0
        viewmat[1, 3] = cv2.getTrackbarPos("Y", "Viewer") / 100.0
        viewmat[2, 3] = cv2.getTrackbarPos("Z", "Viewer") / 100.0
        output, alphas, meta = rasterization(
            means,
            quats,
            scales * cv2.getTrackbarPos("Scaling", "Viewer") / 100.0,
            opacities,
            colors,
            viewmat[None],
            K[None],
            width=width,
            height=height,
            sh_degree=3,
        )

        output_cv = torch_to_cv(output[0])
        if use_checkerboard_background:
            alphas = alphas[0].cpu().numpy()
            output_cv = output_cv.astype(float) * alphas + create_checkerboard(width, height).astype(float) * (1 - alphas)
            output_cv = np.clip(output_cv, 0, 255).astype(np.uint8)
        if show_anaglyph:
            left = output_cv.copy()
            left[..., :2] = 0
            viewmat[:, 3] -= 0.1
            output, _, _ = rasterization(
                means,
                quats,
                scales * cv2.getTrackbarPos("Scaling", "Viewer") / 100.0,
                opacities,
                colors,
                viewmat[None],
                K[None],
                width=width,
                height=height,
                sh_degree=3,
            )
            right = torch_to_cv(output[0])
            if use_checkerboard_background:
                right = right.astype(float) * alphas + create_checkerboard(width, height).astype(float) * (1 - alphas)
                right = np.clip(right, 0, 255).astype(np.uint8)
            right[..., -1] = 0
            output_cv = left + right

        cv2.imshow("Viewer", output_cv)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("3"):
            show_anaglyph = not show_anaglyph
        if key in [ord("w"), ord("a"), ord("s"), ord("d")]:
            if key == ord("w"):
                viewmat[2, 3] -= 0.1
            if key == ord("s"):
                viewmat[2, 3] += 0.1
            if key == ord("a"):
                viewmat[0, 3] += 0.1
            if key == ord("d"):
                viewmat[0, 3] -= 0.1
            update_trackbars_from_viewmat(viewmat)


def torch_to_cv(tensor, permute=False):
    if permute:
        tensor = torch.clamp(tensor.permute(1, 2, 0), 0, 1).cpu().numpy()
    else:
        tensor = torch.clamp(tensor, 0, 1).cpu().numpy()
    return (tensor * 255).astype(np.uint8)[..., ::-1]


if __name__ == "__main__":
    tyro.cli(main)
