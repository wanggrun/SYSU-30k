# Train
# CUDA_VISIBLE_DEVICES=1,2  python3 main.py --datadir /data1/wangguangrun/dataset/market1501/ --bagid 2 --batchid 16 --batchtest 32 --test_every 100 --epochs 300 --decay_type step_250_290 --loss 1*CrossEntropy+2*Triplet   --margin 1.2 --save adam_weak_market           --nGPU 2  --lr 2e-4 --optimizer ADAM --random_erasing --reset --re_rank --amsgrad

# test
# CUDA_VISIBLE_DEVICES=1,2  python3 main.py --datadir /data1/wangguangrun/dataset/market1501/ --bagid 2 --batchid 16 --batchtest 32 --test_every 100 --epochs 300 --decay_type step_250_290 --loss 1*CrossEntropy   --margin 1.2 --save adam_test --nGPU 2  --lr 2e-4 --optimizer ADAM --random_erasing  --re_rank --amsgrad  --test_only --pre_train /home/wangguangrun/pytorch-image-model/MGN-pytorch/work_dirs/adam_weak_hard_cse_tri/model/model_latest.pt

