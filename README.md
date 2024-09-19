# Gradient-Driven 3D Segmentation and Affordance Transfer in Gaussian Splatting Using 2D Masks

This repository contains the code for the paper **Gradient-Driven 3D Segmentation and Affordance Transfer in Gaussian Splatting Using 2D Masks**.

Project page: https://jojijoseph.github.io/3dgs-segmentation
Preprint: https://arxiv.org/abs/2409.11681

## Demo

```bash
python demo.py --help
```

If you need, sample data (chair) can be found at https://drive.google.com/file/d/17xugq_6IaZBpm9B9QYU82hcwBelRR4vh/view?usp=sharing



https://github.com/user-attachments/assets/62f537ca-87e8-4de8-af5d-150ea22dd1ec


## Affordance Transfer

```bash
python affordance_transfer_pipeline.py --help
```

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
