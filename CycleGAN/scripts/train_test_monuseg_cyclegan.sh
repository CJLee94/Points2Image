python train.py --dataroot ~/redwood_mnt/processed_data/MoNuSeg_train.h5 \
--dataset_mode monuseg \
--name cyclegan_color --model cycle_gan --pool_size 50 --no_dropout \
--phase train --preprocess crop --crop_size 256 --batch_size 1 --lambda_identity 0

python train.py --dataroot ~/redwood_mnt/processed_data/MoNuSeg_v2_train.h5 \
--dataset_mode monuseg \
--name cyclegan_gen_color --model cycle_gan --pool_size 50 --no_dropout \
--phase train --preprocess crop --crop_size 256 --batch_size 1 --lambda_identity 0 \
--n_epochs 300 --n_epochs_decay 300 --input_nc 3 

# train with v3
python train.py --dataroot ~/redwood_research/processed_data/MoNuSeg_train_v3.h5 \
--dataset_mode monuseg --name cyclegan_gen_v3_color --model cycle_gan \
--pool_size 50 --no_dropout --phase train --preprocess crop \
--crop_size 256 --batch_size 1 --lambda_identity 0 --n_epochs 500 \
--n_epochs_decay 300 --input_nc 3

# testing script 
python test.py --dataroot ~/redwood_research/processed_data/MoNuSeg_test_v3.h5 \
--dataset_mode monuseg \
--name cyclegan_gen_v3_color --model cycle_gan --phase test --no_dropout \
--preprocess '' --input_nc 3 

python test.py --dataroot ~/redwood_research/processed_data/MoNuSeg_train_v3.h5 \
--dataset_mode monuseg \
--name cyclegan_gen_v3_color --model cycle_gan --phase test --no_dropout \
--preprocess '' --input_nc 3 