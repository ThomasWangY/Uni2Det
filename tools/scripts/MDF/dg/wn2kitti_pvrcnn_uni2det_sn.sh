sh scripts/MDF/dist_train_mdf.sh 8 --cfg_file cfgs/MDF/dg/waymo_nusc_pvrcnn_a5_fade_n_5ep_sn.yaml --source_one_name waymo
sh scripts/MDF/dist_test_mdf.sh 8 --cfg_file cfgs/MDF/dg/kitti_test_pvrcnn.yaml --ckpt ../output/MDF/dg/waymo_nusc_pvrcnn_a5_fade_n_5ep_sn/default/ckpt/checkpoint_epoch_30.pth --source_one_name kitti --source_1 1

cd /root/paddlejob/workspace/env_run/burning/paddle_burning
sh run.sh