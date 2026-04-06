# DSERT-RoLL-Dataset
[CVPR 2026] DSERT-RoLL: Robust Multi-Modal Perception for Diverse Driving Conditions with Stereo Event-RGB-Thermal Cameras, 4D Radar, and Dual-LiDAR

**Project Page:** https://jeongyh98.github.io/dsert-roll/

## Authors

**[Hoonhee Cho](https://chohoonhee.github.io/hoonheecho/)<sup>*</sup>**, **[Jae-Young Kang](https://mickeykang16.github.io/)<sup>*</sup>**, **[Yuhwan Jeong](https://jeongyh98.github.io/)<sup>*</sup>**, **[Yunseo Yang](https://jeongyh98.github.io/dsert-roll/)**, **[Wonyoung Lee](https://jeongyh98.github.io/dsert-roll/)**, **[Youngho Kim](https://jeongyh98.github.io/dsert-roll/)**, **[Kuk-Jin Yoon](https://jeongyh98.github.io/dsert-roll/)**

[VILAB](https://vi.kaist.ac.kr/), KAIST

<sup>*</sup> Equal contribution

## Dataset

### 🤗 Download on Hugging Face

[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/jeongyh98/DSERT-RoLL)

You can download the DSERT-RoLL dataset here:

👉 https://huggingface.co/datasets/jeongyh98/DSERT-RoLL

**Total Size:** 269 GB

### Environmental Conditions

- **Weather:** Clear / Fog / Light Rain / Heavy Rain / Light Snow / Heavy Snow
- **Lighting:** Normal / Low Light / Over Expose / HDR

### Train/Validation Split

| Split | Sequences |
|------:|----------:|
| Train | 132 |
| Val   | 58  |
| Total | 190 |

### Directory Structure

```text
DSERT-RoLL/
│
├── split/
│   ├── train.txt
│   └── val.txt
│
├── {weather}/                        # Clear | Fog | Heavy_Rain | Heavy_Snow | Light_Rain | Light_Snow
│   └── {sequence_name}/              # e.g. 2025_07_16_21_36_14
│       ├── label.pkl                 # annotations & calibration for the sequence
│       │
│       ├── rectified_crop_RGB_L/     # Left RGB images         (.jpg)
│       ├── rectified_crop_RGB_R/     # Right RGB images        (.jpg)
│       ├── rectified_EVENT_L/        # Left event frames       (.npz)
│       ├── rectified_EVENT_R/        # Right event frames      (.npz)
│       ├── rectified_THERMAL_L/      # Left thermal images     (.npz)
│       ├── rectified_THERMAL_R/      # Right thermal images    (.npz)
│       ├── LIDAR_LIVOX_Tilted/       # Livox LiDAR point cloud (.npy)
│       ├── LIDAR_OUSTER_Tilted_90_degree/  # Ouster LiDAR point cloud (.npy)
│       └── RADAR_Tilted/             # Radar point cloud       (.npy)
```

### Statistics

```text
Clear:       102 sequences / 11,877 frames
Fog:          32 sequences /  3,282 frames
Heavy_Rain:   18 sequences /  1,898 frames
Light_Rain:   17 sequences /  2,214 frames
Light_Snow:   14 sequences /  1,428 frames
Heavy_Snow:    7 sequences /    980 frames
─────────────────────────────────────────
Total:       190 sequences / 21,679 frames
```

### `label.pkl` Structure

```text
label.pkl
├── meta
│   ├── weather          # weather condition
│   ├── light            # lighting condition (e.g. Low_Light)
│   ├── sequence_len     # number of frames
│   └── calibration      # sensor-wise intrinsic/extrinsic matrices
│       ├── Livox, Ouster, Radar
│       └── RGB_L/R, Thermal_L/R, Event_L/R
│
└── info                 # per-frame list
  ├── time_stamp
  ├── sample_idx / frame_idx
  ├── sequence_name
  ├── pose
  ├── sensor           # relative file paths for each sensor
  │   ├── rgb_left_path / rgb_right_path
  │   ├── event_left_path / event_right_path
  │   ├── thermal_left_path / thermal_right_path
  │   ├── livox_path / ouster_path
  │   └── radar_path
  └── annos            # 3D bounding box annotations
    ├── name, obj_ids
    ├── dimensions, location, heading_angles
    └── gt_boxes_livox / gt_boxes_rgb / gt_boxes_event / gt_boxes_thermal
```

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{cho2026dsertroll,
  title={DSERT-RoLL: Robust Multi-Modal Perception for Diverse Driving Conditions with Stereo Event-RGB-Thermal Cameras, 4D Radar, and Dual-LiDAR},
  author={Cho, Hoonhee and Kang, Jae-Young and Jeong, Yuhwan and Yang, Yunseo and Lee, Wonyoung and Kim, Youngho and Yoon, Kuk-Jin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
