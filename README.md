# FC-GRMA
Code for "Enhanced hybrid prototype for few-shot class-incremental gait recognition in multi-activity scenarios using wearable sensors".

# Training Scripts
## usc-had
pre-train
```
python train.py --project 'base' -dataset 'usc-had' -epochs_base 60 -lr_base 0.01 -step 20 -gamma 0.1 -episode_way 6 -model_dir None
```
meta-train
```
python train.py --project 'meta' -dataset 'usc-had' -epochs_base 60 -lr_base 0.001 -lrg1 0.0001 -lrg2 0.0001 -step 20 -gamma 0.5 -episode_way 6 -model_dir 'models/usc-had/base/session0_max_acc.pth'
```

# Acknowledgment
Our project implementation references the following repositories. Thanks to the author for the open source code.
- [CEC][CVPR2021]
