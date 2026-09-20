# BEVine

Official code repository for our accepted paper in Smart Agriculture Technology "Early Fusion Multi-View Aggregation for Multi-Camera Cattle Tracking"

## Abstract 

Intensive dairy farming operations require automated monitoring solutions to efficiently manage large herds across expansive areas. However, existing approaches face significant limitations. Single-camera systems provide insufficient coverage due to blind spots and reduced spatial resolution at greater distances, while current multi-camera tracking methods depend on detecting animals in individual views before cross-camera association, and are often restricted to specific barns or breeds. We introduce BEVine (bird's eye view for bovine tracking), a novel open-source multi-camera tracking framework that performs early multi-view aggregation by detecting animals directly in a unified bird's eye view (BEV) representation, rather than associating detections across separate camera views. Our practical visual localisation pipeline generates BEV ground-truth positions from time-synchronised multi-camera footage, supported by a web-based user interface for annotation refinement. We introduce a multi-sequence training protocol that prevents scene-specific overfitting of the temporal BEV feature cache, and two complementary architectural extensions: per-camera image auxiliary supervision providing explicit foot-point and bounding box geometry, and a differentiable calibration refinement module that learns per-camera extrinsic corrections end-to-end. We demonstrate robust performance across two distinct farm datasets with varying camera configurations and cattle breeds: our JerCCows dataset (8 cameras, Jersey cattle) and the publicly available MmCows dataset (4 cameras, Holstein cattle), achieving multi-object tracking accuracies of 84.6% and 85.7%, respectively. These results establish early fusion BEV tracking as a viable and scalable solution for precision livestock farming across diverse agricultural settings. The code is available at https://github.com/MahejabeenNidhi/BEVine

## Dataset annotation

### How to annotate your own 3D dataset

You can now annotate your own 3D bounding boxes on multi-camera systems

```
python annotate3d.py --dataset /path/to/mmcows_images \
    --calibration_json output_barn_multi/camera_calibration.json \
    --floorplan_json output_barn_multi/floorplan.json \
    --review_dir output_annotations/mmcows
```
<img width="5809" height="2729" alt="Figure1" src="https://github.com/user-attachments/assets/902d3827-ad64-454b-81d3-b5fc7badf981" />


### How to annotate your own 2D dataset

Our dataset contains time-synchronised images from eight cameras. 

The undistorted images were annotated using LabelMe. All the images were annotated with bounding boxes with label names in the format {action}_{ID}. The actions are limited to "standing", "lying", and "feeding". The IDs were random numbers from 1 to the maximum number of cows in that timestamp. Some examples include standing_1, lying_2, and feeding_3. It is critical to ensure that the action and the ID are consistent for the same animal across the camera views.

Once the images are annotated, we can use the WorldTrack/localisation_tools/visual_localization.py script to output annotations of the images suitable for TrackTacular, which is similar to the WildTrack dataset. 

The JSON output files from using the visual_localization.py script that be further modified using our web-based annotation tool BEVineAnnotationTool.html which will work on a browser (only tested on Google Chrome). 

[![Watch the video](https://img.youtube.com/vi/D_FNVcT1D2U/maxresdefault.jpg)](https://youtu.be/D_FNVcT1D2U)


## How to train 

```
python world_track.py fit   -c configs/t_fit.yml   -c configs/d_{dataset_config}.yml   -c configs/m_bevformer.yml

# Example: training mmcows (we have only developed and tested BEVine on BEVFormer)

python world_track.py fit   -c configs/t_fit.yml   -c configs/d_mmcows_multiseq.yml   -c configs/m_bevformer.yml
```

## How to test 

```
python world_track.py test     -c configs/t_fit.yml     -c configs/d_{dataset_test}.yml     -c configs/m_bevformer.yml     --ckpt path/to/checkpoint/last.ckpt

# Example

python world_track.py test     -c configs/t_fit.yml     -c configs/d_mmcows_multiseq.yml     -c configs/m_bevformer.yml     --ckpt_path lightning_logs/MmCows_Ablation_X/checkpoints/last.ckpt
```

## MmCows Dataset 

You can download the images from the original data repository for MmCows and organise them accordingly in Image_subset folders to run BEVine

Our annotations on the MmCows dataset can be downloaded from [here](https://drive.google.com/drive/folders/1cwHKAYhcS3lYNl5rutRsW-CcCnqV4mDc?usp=sharing)

The training weight and the refined calibration of the full BEVine model can be downloaded from [here](https://drive.google.com/drive/folders/1mw0NHEG_sX161I5D68iiTTgGZxTSnUyL?usp=sharing)

## JerCCows Dataset

This dataset will be made available upon manuscript acceptance and available upon reasonable request. 

## Acknowledgements

This work is built upon the [TrackTacular](https://github.com/tteepe/TrackTacular) codebase by Teepe et al. We gratefully acknowledge their foundational work on lifting multi-view detection and tracking to the Bird's Eye View.

This codebase can be tested on the [MmCows](https://github.com/neis-lab/mmcows) dataset, a multimodal dataset for dairy cattle monitoring.

Our dataset is now made publicly available on [hugging face](https://huggingface.co/datasets/MaeNidhi/JerCCows)

## Citation

If you find BEVine useful for your research, please cite our work:

```bibtex
Nidhi, M.H., Guo, C., Lyu, L., He, Z., Guo, Z., Liu, K., Flay, K.J., 2026.
Early Fusion Multi-View Aggregation for Multi-Camera Cattle Tracking.
Smart Agricultural Technology 102473. https://doi.org/10.1016/j.atech.2026.102473
