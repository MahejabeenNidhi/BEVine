# BEVine
Multi-Camera Tracking of Cattle Herds

## Dataset annotation

### Formatting mmCows dataset for use

```

```

### How to annotate your own dataset

Our dataset contains time-synchronised images from eight cameras. 

The undistorted images were annotated using LabelMe. All the images were annotated with bounding boxes with label names in the format {action}_{ID}. The actions are limited to "standing", "lying", and "feeding". The IDs were random numbers from 1 to the maximum number of cows in that timestamp. Some examples include standing_1, lying_2, and feeding_3. It is critical to ensure that the action and the ID are consistent for the same animal across the camera views.

Once the images are annotated, we can use the visual_localization_IE_TT.py script to output annotations of the images suitable for TrackTacular - which is similar to the WildTrack dataset. 

### How 

