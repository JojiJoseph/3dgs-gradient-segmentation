# Gradient-Driven 3D Segmentation and Affordance Transfer in Gaussian Splatting Using 2D Masks

This repository contains the code for the paper **Gradient-Driven 3D Segmentation and Affordance Transfer in Gaussian Splatting Using 2D Masks**.

Project page: https://jojijoseph.github.io/3dgs-segmentation
Preprint: https://arxiv.org/abs/2409.11681

**Update**: Sep 24, 2024 - Our poster titled **Segmentation of 3D Gaussians using Masked Gradients**, corresponds to the **preliminary work**, has been accepted to **SIGGRAPH Asia 2024**.

## Setup

Please install the dependencies listed in `requirements.txt` via `pip install -r requirements.txt`. Download `sam2_hiera_large.pt` from https://huggingface.co/facebook/sam2-hiera-large/tree/main and place it in the `./checkpoints` folder. 

Other than that, it's a self-contained repo. Please feel free to raise an issue if you face any problems while running the code.

## Demo

```bash
python demo.py --help
```

If needed, sample data (chair) can be found [here](https://drive.google.com/file/d/17xugq_6IaZBpm9B9QYU82hcwBelRR4vh/view?usp=sharing). Please create a folder named `data` on root folder and extract the contents of zip file to that folder. Then simply run `python demo.py`.



https://github.com/user-attachments/assets/62f537ca-87e8-4de8-af5d-150ea22dd1ec


## Affordance Transfer

```bash
python affordance_transfer_pipeline.py --help
```

Left: Source images, Middle: 2D-2D affordance transfer, Right: 2D-3D Affordance transfer

https://github.com/user-attachments/assets/65406bb7-f690-42d5-aca6-59046e08de08


## Affordance Transfer - Evaluation

Download trained scenes from [here](https://drive.google.com/file/d/1-f-rW3U1H5RqdCvp-1BcuSZxrEGc3Rxo/view?usp=sharing). Original scenes (without trained Gaussian Splat models) can be found at https://users.umiacs.umd.edu/~fer/affordance/Affordance.html.

```sh
sh eval_affordance_transfer.sh | tee affordance_transfer.log
```


## Some Downstream Applications

Augmented reality.

https://github.com/user-attachments/assets/20ee5c8b-031e-423d-890d-368e1a9c5731

Reorganzing objects in real time.

https://github.com/user-attachments/assets/91cc6ef1-0fd2-44a5-8881-61a042662a95

## Acknowledgements

A big thanks to the following tools/libraries, which were instrumental in this project:

- [gsplat](https://github.com/nerfstudio-project/gsplat): 3DGS rasterizer.
- [SAM 2](https://github.com/facebookresearch/segment-anything-2): To track masks throughout the frames.
- [YOLO-World](https://github.com/AILab-CVC/YOLO-World) via [ultralytics](https://docs.ultralytics.com/models/yolo-world/): To find Initial bounding box.
- [labelme](https://github.com/labelmeai/labelme): To label the source images for affordance transfer.

## Citation
If you find this paper or the code helpful for your work, please consider citing our preprint,
```
@article{joji2024gradient,
  title={Gradient-Driven 3D Segmentation and Affordance Transfer in Gaussian Splatting from 2D Masks},
  author={Joji Joseph and Bharadwaj Amrutur and Shalabh Bhatnagar},
  journal={arXiv preprint arXiv:2409.11681},
  year={2024},
  url={https://arxiv.org/abs/2409.11681}
}
```
