sh scripts/MDF/dist_train_mdf.sh 8 --cfg_file cfgs/MDF/dg/waymo_nusc_voxel_rcnn_a1_fade_n_5ep.yaml --source_one_name waymo
sh scripts/MDF/dist_test_mdf.sh 8 --cfg_file cfgs/MDF/dg/kitti_test_voxel_rcnn.yaml --ckpt ../output/MDF/dg/waymo_nusc_voxel_rcnn_a1_fade_n_5ep/default/ckpt/checkpoint_epoch_30.pth --source_one_name kitti --source_1 1
