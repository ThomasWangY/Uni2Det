sh scripts/MDF/dist_train_mdf_3db.sh 8 --cfg_file ./cfgs/MDF/KNW/knw_voxel_rcnn_feat_3_uni2det.yaml 
sh scripts/MDF/dist_test_mdf_3db.sh 8 --cfg_file ./cfgs/MDF/KNW/knw_voxel_rcnn_feat_3_uni2det.yaml --ckpt ../output/cfgs/MDF/KNW/knw_voxel_rcnn_feat_3_uni2det/default/ckpt/checkpoint_epoch_30.pth --source_1 1
sh scripts/MDF/dist_test_mdf_3db.sh 8 --cfg_file ./cfgs/MDF/KNW/knw_voxel_rcnn_feat_3_uni2det.yaml --ckpt ../output/cfgs/MDF/KNW/knw_voxel_rcnn_feat_3_uni2det/default/ckpt/checkpoint_epoch_30.pth --source_1 2
sh scripts/MDF/dist_test_mdf_3db.sh 8 --cfg_file ./cfgs/MDF/KNW/knw_voxel_rcnn_feat_3_uni2det.yaml --ckpt ../output/cfgs/MDF/KNW/knw_voxel_rcnn_feat_3_uni2det/default/ckpt/checkpoint_epoch_30.pth --source_1 3
