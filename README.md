# BEVine3D

Introducing 3D bounding box integration

## Usage

### Training
```
python world_track.py fit \
  --config configs/t_fit.yml \
  --config configs/m_bevformer.yml \
  --config configs/d_mmcows_multiseq.yml \
  --data.init_args.annotation_mode 3d \
  --model.swin_pretrained_path "/path/to/exported_last.pt" \
```
### Testing
```
python world_track.py test \
  --config configs/t_fit.yml \
  --config configs/m_bevformer.yml \
  --config configs/d_mmcows_multiseq.yml \
  --ckpt_path /path/to/last.ckpt \
  --data.init_args.annotation_mode 3d \
  --model.swin_pretrained_path "/path/to/exported_last.pt" \
```
To switch between 2d and 3d training, change the annotation_mode

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
