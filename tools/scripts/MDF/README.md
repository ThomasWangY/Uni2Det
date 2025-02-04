# train and test with Uni^2Det (under './scripts/MDF/uni2det')
sh ./scripts/MDF/uni2det/dist_train_pv_nk.sh
sh ./scripts/MDF/uni2det/dist_train_pv_wk.sh
sh ./scripts/MDF/uni2det/dist_train_pv_wn.sh
sh ./scripts/MDF/uni2det/dist_train_voxel_nk.sh
sh ./scripts/MDF/uni2det/dist_train_voxel_wk.sh
sh ./scripts/MDF/uni2det/dist_train_voxel_wn.sh

# conduct zero-shot learning with Uni^2Det (under './scripts/MDF/dg')
## 1. train sn on single dataset
sh ./scripts/MDF/dg/nusc2kitti_pvrcnn_sn.sh
sh ./scripts/MDF/dg/waymo2kitti_pvrcnn_sn.sh
## 2. train uni2det on multiple datasets
sh ./scripts/MDF/dg/wn2kitti_pvrcnn_uni2det.sh
sh ./scripts/MDF/dg/wn2kitti_voxel_rcnn_uni2det.sh
## 3. train uni2det with sn on multiple datasets
sh ./scripts/MDF/dg/wn2kitti_pvrcnn_uni2det_sn.sh
sh ./scripts/MDF/dg/wn2kitti_voxel_rcnn_uni2det_sn.sh

# train and test on 3 datasets (under './scripts/MDF/db_3')
## 1. train with uni2det
sh ./scripts/MDF/db_3/dist_train_voxel_db_3_uni2det.sh
## 2. train with uni3d
sh ./scripts/MDF/db_3/dist_train_voxel_db_3_uni3d.sh

# train and test with baseline and with SeNet (under './scripts/MDF/new')
sh ./scripts/MDF/new/dist_train_voxel_baseline.sh
sh ./scripts/MDF/new/dist_train_voxel_se.sh

# export PATH=/root/paddlejob/workspace/env_run/wyb/conda/bin:$PATH
# ps -ef | grep train.py | grep -v grep | awk '{print $2}' | xargs kill -9
# ps -ef | grep run.py | grep -v grep | awk '{print $2}' | xargs kill -9
# ps -ef | grep train_multi_db.py | grep -v grep | awk '{print $2}' | xargs kill -9
# ps -ef | grep test_multi_db.py | grep -v grep | awk '{print $2}' | xargs kill -9
