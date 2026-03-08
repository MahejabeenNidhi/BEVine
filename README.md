# BEVine
Multi-Camera Tracking of Cattle Herds

## Dataset annotation

### How to annotate your own dataset

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

