# Training & Evaluation

## Zero-shot Evaluation on Unseen Datasets

* Train with a single dataset and evaluate on KITTI in the zero-shot setting on the **Car** category with SN.

```shell script
# e.g. nusc -> kitti on pv-rcnn

# Training
sh scripts/MDF/dist_train.sh 8 --cfg_file cfgs/MDF/dg/only_nusc/pvrcnn_old_anchor_sn.yaml

# Evaluation
sh scripts/MDF/dist_test.sh 8 --cfg_file cfgs/MDF/dg/kitti_test_pvrcnn.yaml --ckpt ../output/MDF/dg/only_nusc/pvrcnn_old_anchor_sn/default/ckpt/checkpoint_epoch_50.pth
```

* Train with two datasets (nuScenes and Waymo) and evaluate on KITTI in the zero-shot setting on the **Car** category without SN.

```shell script
# e.g. nusc + waymo -> kitti on voxel-rcnn (without SN)

# Training
sh scripts/MDF/dist_train_mdf.sh 8 --cfg_file cfgs/MDF/dg/waymo_nusc_voxel_rcnn_a1_fade_n_5ep.yaml --source_one_name waymo

# Evaluation
sh scripts/MDF/dist_test_mdf.sh 8 --cfg_file cfgs/MDF/dg/kitti_test_voxel_rcnn.yaml --ckpt ../output/MDF/dg/waymo_nusc_voxel_rcnn_a1_fade_n_5ep/default/ckpt/checkpoint_epoch_30.pth --source_one_name kitti --source_1 1
```

- Train with two datasets (nuScenes and Waymo) and evaluate on KITTI in the zero-shot setting on the **Car** category with SN.

```shell script
# e.g. nusc + waymo -> kitti on voxel-rcnn (with SN)

# Training
sh scripts/MDF/dist_train_mdf.sh 8 --cfg_file cfgs/MDF/dg/waymo_nusc_voxel_rcnn_a1_fade_n_5ep_sn.yaml --source_one_name waymo

# Evaluation
sh scripts/MDF/dist_test_mdf.sh 8 --cfg_file cfgs/MDF/dg/kitti_test_voxel_rcnn.yaml --ckpt ../output/MDF/dg/waymo_nusc_voxel_rcnn_a1_fade_n_5ep_sn/default/ckpt/checkpoint_epoch_30.pth --source_one_name kitti --source_1 1
```

* We also provide other merged training and evaluation scripts under `tools\scripts\MDF\dg` for various settings.

## Multi-Dateset 3D Object Detection

* Train with two datasets and evaluate on both datasets.
```shell script
# e.g. nusc + kitti on pv-rcnn

# Training
sh scripts/MDF/dist_train_mdf.sh 8 --cfg_file ./cfgs/MDF/nusc_kitti/uni2det/nusc_kitti_pvrcnn_feat_3_a5_fade2.yaml --source_one_name kitti

# Evaluation on kitti
sh scripts/MDF/dist_test_mdf.sh 8 --cfg_file ./cfgs/MDF/nusc_kitti/uni2det/nusc_kitti_pvrcnn_feat_3_a5_fade2.yaml --ckpt ../output/cfgs/MDF/nusc_kitti/uni2det/nusc_kitti_pvrcnn_feat_3_a5_fade2/default/ckpt/checkpoint_epoch_30.pth --source_one_name kitti --source_1 1

# Evaluation on nuscenes
sh scripts/MDF/dist_test_mdf.sh 8 --cfg_file ./cfgs/MDF/nusc_kitti/uni2det/nusc_kitti_pvrcnn_feat_3_a5_fade2.yaml --ckpt ../output/cfgs/MDF/nusc_kitti/uni2det/nusc_kitti_pvrcnn_feat_3_a5_fade2/default/ckpt/checkpoint_epoch_30.pth --source_one_name kitti --source_1 2
```

* We also provide other merged training and evaluation scripts under `tools\scripts\MDF\uni2det` for various combinations.
## Joint training on Waymo-KITTI-nuScenes Consolidations

* Train with three datasets and evaluate on all datasets.
```shell script
# e.g. nusc + kitti + waymo on voxel-rcnn

# Training
sh scripts/MDF/dist_train_mdf_3db.sh 8 --cfg_file ./cfgs/MDF/KNW/knw_voxel_rcnn_feat_3_uni2det.yaml 

# Evaluation on waymo
sh scripts/MDF/dist_test_mdf_3db.sh 8 --cfg_file ./cfgs/MDF/KNW/knw_voxel_rcnn_feat_3_uni2det.yaml --ckpt ../output/cfgs/MDF/KNW/knw_voxel_rcnn_feat_3_uni2det/default/ckpt/checkpoint_epoch_30.pth --source_1 1

# Evaluation on kitti
sh scripts/MDF/dist_test_mdf_3db.sh 8 --cfg_file ./cfgs/MDF/KNW/knw_voxel_rcnn_feat_3_uni2det.yaml --ckpt ../output/cfgs/MDF/KNW/knw_voxel_rcnn_feat_3_uni2det/default/ckpt/checkpoint_epoch_30.pth --source_1 2

# Evaluation on nuscenes
sh scripts/MDF/dist_test_mdf_3db.sh 8 --cfg_file ./cfgs/MDF/KNW/knw_voxel_rcnn_feat_3_uni2det.yaml --ckpt ../output/cfgs/MDF/KNW/knw_voxel_rcnn_feat_3_uni2det/default/ckpt/checkpoint_epoch_30.pth --source_1 3
```

* We also provide other merged training and evaluation scripts under `tools\scripts\MDF\db_3` for various settings.