sh scripts/MDF/dist_train_mdf.sh 8 --cfg_file ./cfgs/MDF/waymo_nusc/uni2det/waymo_nusc_voxel_rcnn_feat_3_a1_fade_n_5ep.yaml --source_one_name waymo
sh scripts/MDF/dist_test_mdf.sh 8 --cfg_file ./cfgs/MDF/waymo_nusc/uni2det/waymo_nusc_voxel_rcnn_feat_3_a1_fade_n_5ep.yaml --ckpt ../output/cfgs/MDF/waymo_nusc/uni2det/waymo_nusc_voxel_rcnn_feat_3_a1_fade_n_5ep/default/ckpt/checkpoint_epoch_30.pth --source_one_name waymo --source_1 1
sh scripts/MDF/dist_test_mdf.sh 8 --cfg_file ./cfgs/MDF/waymo_nusc/uni2det/waymo_nusc_voxel_rcnn_feat_3_a1_fade_n_5ep.yaml --ckpt ../output/cfgs/MDF/waymo_nusc/uni2det/waymo_nusc_voxel_rcnn_feat_3_a1_fade_n_5ep/default/ckpt/checkpoint_epoch_30.pth --source_one_name waymo --source_1 2

cd /root/paddlejob/workspace/env_run/burning/paddle_burning
sh run.sh