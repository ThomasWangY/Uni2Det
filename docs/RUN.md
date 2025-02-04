# Training & Evaluation

## Zero-shot Evaluation on Unseen Datasets

### Training stage 

* Train with consistent point-cloud range (employing Waymo range) and C.A. (Coordinate-origin Alignment) using multiple GPUs

```shell script
sh scripts/MDF/dist_train_mdf.sh ${NUM_GPUs} \
--cfg_file ./cfgs/MDF/waymo_nusc/waymo_nusc_pvrcnn_feat_3_uni3d.yaml \
--source_one_name waymo
```

* Train with consistent point-cloud range (employing Waymo range) and C.A. (Coordinate-origin Alignment) using multiple machines

```shell script
sh scripts/MDF/slurm_train_multi_db.sh ${PARTITION} ${JOB_NAME} ${NUM_NODES} \
--cfg_file ./cfgs/MDF/waymo_nusc/waymo_nusc_pvrcnn_feat_3_uni3d.yaml \
--source_one_name waymo
```

* Train other baseline detectors such as PV-RCNN++ using multiple GPUs

```shell script
sh scripts/MDF/dist_train_mdf.sh ${NUM_GPUs} \
--cfg_file ./cfgs/MDF/waymo_nusc/waymo_nusc_pvplus_feat_3_uni3d.yaml \
--source_one_name waymo
```

* Train other baseline detectors such as Voxel-RCNN using multiple GPUs

```shell script
sh scripts/MDF/dist_train_mdf.sh ${NUM_GPUs} \
--cfg_file ./cfgs/MDF/waymo_nusc/waymo_nusc_voxel_rcnn_feat_3_uni3d.yaml \
--source_one_name waymo&ensp;&ensp;
```

### Evaluation stage

* Note that for the KITTI-related evaluation, please try --set DATA_CONFIG.FOV_POINTS_ONLY True to enable front view point cloud only. We report the best results on KITTI for testing all epochs on the validation set.

  - ${FIRST_DB_NAME} denotes that the fisrt dataset name of the merged two dataset, which is used to split the merged dataset into two individual datasets.

  - ${DB_SOURCE} denotes the dataset to be tested.


* Test the models using multiple GPUs

```shell script
sh scripts/MDF/dist_test_mdf.sh ${NUM_GPUs} \
--cfg_file ${CFG_FILE} \
--ckpt ${CKPT} \
--source_one_name ${FIRST_DB_NAME} \
--source_1 ${DB_SOURCE} 
```

* Test the models using multiple machines

## Multi-Dateset 3D Object Detection

### Training stage 

* Train with consistent point-cloud range (employing Waymo range) and C.A. (Coordinate-origin Alignment) using multiple GPUs
```shell script
sh scripts/MDF/dist_train_mdf.sh ${NUM_GPUs} \
--cfg_file ./cfgs/MDF/waymo_nusc/waymo_nusc_pvrcnn_feat_3_uni3d.yaml \
--source_one_name waymo
```

* Train with consistent point-cloud range (employing Waymo range) and C.A. (Coordinate-origin Alignment) using multiple machines
```shell script
sh scripts/MDF/slurm_train_multi_db.sh ${PARTITION} ${JOB_NAME} ${NUM_NODES} \
--cfg_file ./cfgs/MDF/waymo_nusc/waymo_nusc_pvrcnn_feat_3_uni3d.yaml \
--source_one_name waymo
```

* Train other baseline detectors such as PV-RCNN++ using multiple GPUs
```shell script
sh scripts/MDF/dist_train_mdf.sh ${NUM_GPUs} \
--cfg_file ./cfgs/MDF/waymo_nusc/waymo_nusc_pvplus_feat_3_uni3d.yaml \
--source_one_name waymo
```

* Train other baseline detectors such as Voxel-RCNN using multiple GPUs
```shell script
sh scripts/MDF/dist_train_mdf.sh ${NUM_GPUs} \
--cfg_file ./cfgs/MDF/waymo_nusc/waymo_nusc_voxel_rcnn_feat_3_uni3d.yaml \
--source_one_name waymo
```

### Evaluation stage
* Note that for the KITTI-related evaluation, please try --set DATA_CONFIG.FOV_POINTS_ONLY True to enable front view point cloud only. We report the best results on KITTI for testing all epochs on the validation set.

    - ${FIRST_DB_NAME} denotes that the fisrt dataset name of the merged two dataset, which is used to split the merged dataset into two individual datasets.

    - ${DB_SOURCE} denotes the dataset to be tested.


* Test the models using multiple GPUs
```shell script
sh scripts/MDF/dist_test_mdf.sh ${NUM_GPUs} \
--cfg_file ${CFG_FILE} \
--ckpt ${CKPT} \
--source_one_name ${FIRST_DB_NAME} \
--source_1 ${DB_SOURCE} 
```

* Test the models using multiple machines
```shell script
sh scripts/MDF/slurm_test_mdb_mgpu.sh ${PARTITION} ${JOB_NAME} ${NUM_NODES} \
--cfg_file ${CFG_FILE} \
--ckpt ${CKPT} \
--source_one_name ${FIRST_DB_NAME} \
--source_1 ${DB_SOURCE}
```

## Joint-training on Waymo-KITTI-nuScenes Consolidations

### Training Stage

* Train with consistent point-cloud range (employing Waymo range) using multiple GPUs
```shell script
sh scripts/MDF/dist_train_mdf_3db.sh ${NUM_GPUs} \
--cfg_file ./cfgs/MDF/KNW/knw_voxel_rcnn_feat_3_uni3d.yaml
```

* Train with consistent point-cloud range (employing Waymo range) using multiple machines
```shell script
sh scripts/MDF/slurm_train_multi_db_3db.sh ${PARTITION} ${JOB_NAME} ${NUM_NODES} \
--cfg_file ./cfgs/MDF/KNW/knw_voxel_rcnn_feat_3_uni3d.yaml
```

* Train other baseline detectors such as PV-RCNN using multiple GPUs
```shell script
sh scripts/MDF/dist_train_mdf_3db.sh ${NUM_GPUs} \
--cfg_file ./cfgs/MDF/KNW/knw_pvrcnn_feat_3_uni3d.yaml
```

### Evaluation stage
* ${DB_SOURCE} denotes the dataset to be tested.


* Test the models using multiple GPUs
```shell script
sh scripts/MDF/dist_test_mdf_3db.sh ${NUM_GPUs} \
--cfg_file ${CFG_FILE} \
--ckpt ${CKPT} \
--source_1 ${DB_SOURCE} 
```

* Test the models using multiple machines
```shell script
sh scripts/MDF/slurm_test_mdb_mgpu_3db.sh ${PARTITION} ${JOB_NAME} ${NUM_NODES} \
--cfg_file ${CFG_FILE} \
--ckpt ${CKPT} \
--source_1 ${DB_SOURCE} 
```