cd code
mkdir ../logs
python train_pancreas_3d_ged_supcontrast.py --gpu 0 --with_dice=True --ssn_rank=8 --num_mc_samples=6 --consistency=0.15 --consistency_type=ged --pairwise_dist_metric=cls_mean_dice --with_uncertainty_mask=False  --data_loader_labeled_aug=None --unlabeled_aug_with_resize=False --baseline_noout=False --cross_image_contrast=False --cross_image_sampling=False --random_sampled_num=9000 --sup_cont_weight=0.09 --exp=exp_pancreas_aua_stage1 > ../logs/exp_pancreas_aua_stage1.txt 2>&1 
python test_pancreas_ged_supcontrast_all.py --gpu 0 --model=exp_pancreas_aua_stage1 > ../logs/exp_pancreas_aua_stage1_test_all.txt 2>&1

python train_pancreas_3d_semantic_dist_init.py --gpu=0 --load_model_name=exp_pancreas_aua_stage1 --load_epoch_num=3000 --exp=pancreas_exp_semantic_dist_init_from_pancreas_aua_stage1 --feat_dist_save_name=feat_dist_from_pancreas_aua_stage1 --out_dist_save_name=out_dist_from_pancreas_aua_stage1  

python train_pancreas_3d_pcl.py --gpu 0  --with_dice=True --ssn_rank=8 --data_loader_labeled_aug=None --unlabeled_aug_with_resize=False --load_model_name=exp_pancreas_aua_stage1 --load_epoch_num=3000  --feat_dist_save_name=feat_dist_from_pancreas_aua_stage1 --out_dist_save_name=out_dist_from_pancreas_aua_stage1 --estimator_starts_iter=0 --ORACLE=False --LAMBDA_PSEUDO=1 --LAMBDA_FEAT=0.1 --LAMBDA_OUT=0 --IGNORE_LABEL=0 --exp=exp_pancreas_aua_stage2 > ../logs/exp_pancreas_aua_stage2.txt 2>&1 
python test_pancreas_ged_supcontrast_all.py --gpu 0 --model=exp_pancreas_aua_stage2 > ../logs/exp_pancreas_aua_stage2_test_all.txt 2>&1

