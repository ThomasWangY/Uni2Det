sh scripts/MDF/dist_train.sh 8 --cfg_file cfgs/MDF/dg/only_nusc/pvrcnn_old_anchor_sn.yaml
sh scripts/MDF/dist_test.sh 8 --cfg_file cfgs/MDF/dg/kitti_test_pvrcnn.yaml --ckpt ../output/MDF/dg/only_nusc/pvrcnn_old_anchor_sn/default/ckpt/checkpoint_epoch_50.pth
