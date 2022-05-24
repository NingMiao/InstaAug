# InstaAug
Codes for the paper 'Learning Instance-Specific Data Augmentations'. The main package is `InstaAug_module` and we show how to apply InstaAug to various tasks in `examples`.

## Usage

### Rotation on Mario-Iggy
Change to the mixmo folder:
`cd examples/mixmo-pytorch`.

Train the model:
`python -m scripts.train --config_path config/marioiggy/exp_tinyimagenet_res18_1net_standard_bar1_test.yaml --dataplace ../../data/ --saveplace model_marioiggy/ --gpu 0 --info test --from_scratch --Li_config_path ../../InstaAug_module/configs/config_rotation_supervised.yaml`.

### Cropping on Tiny-Imagenet
Change to the mixmo folder:
`cd examples/mixmo-pytorch`.

First per-train the model with no augmentation: 
`python -m scripts.train --config_path config/tiny/exp_tinyimagenet_res18_1net_standard_bar1_test_pretrain.yaml --dataplace ../../data/ --saveplace model_tiny/ --gpu 5 --info memory --from_scratch --max_tolerance 10`.

Then train with InstaAug:
`python -m scripts.train --config_path config/tiny/exp_tinyimagenet_res18_1net_standard_bar1_test.yaml --dataplace ../../data --saveplace model_tiny/ --gpu 0 --info test --from_scratch --Li_config_path ../../InstaAug_module/configs/config_crop_supervised.yaml --max_tolerance 10 -checkpoint model_tiny/exp_tinyimagenet_res18_1net_testmemory/checkpoint_epoch_010.ckpt`.

### Color-jittering on RawFooT
Change to the mixmo folder:
`cd examples/mixmo-pytorch`.

Train the model:
`python -m scripts.train --config_path config/rawfoot/exp_tinyimagenet_res18_1net_standard_bar1_test_pretrain.yaml --dataplace ../../data/ --saveplace model_rawfoot/ --gpu 0 --info test --from_scratch --Li_config_path ../../InstaAug_module/configs/config_color_jittering_supervised.yaml`

### Contrastive learning on Tiny-Imagenet
Change to the self-supervised folder:
`cd examples/self-supervised`.

Train the model:
`python -m train --dataset tiny_in --epoch 500 --lr 2e-3 --emb 128 --method contrastive--model_folder model/test --Li_config_path ../../InstaAug_module/configs/config_crop_contrastive.yaml--eval_every 50 --crop_s0 1.0 --crop_s1 1.0 --crop_r0 1.0 --crop_r1 1.0 --wandb_name test --entropy_weights 0.003 --num_workers 4 --target_entropy 3.7`


