# InstaAug
This is the official codebase for the paper [**Instance--Specific Augmentation: Capturing Local Invariances**](https://arxiv.org/pdf/2206.00051.pdf). 

--------------------
InstaAug is a method for automatically learning input-specific augmentations from data. This is achieved by introducing an augmentation module that maps an input to a distribution over transformations. As a plug-in module, InstaAug can effeciently work with supervised and unsupervised models to learn and apply input-specific augmentations.


## Usage
The main package is `InstaAug_module` and please train the model through `scripts/train`.
```
scripts/train.py:
  --config_path: config path for the task and dataset 
  --Li_config_path: config path for InstaAug module
  --dataplace: data folder
  --saveplace: model save place 
  --gpu: gpu to use
  --info: checkpoint and log suffix to distinguish between experiments
  --from_scratch: binary flag indicating whether to train from scratch or to load checkpoints
```

We show how to apply InstaAug to various tasks in `examples`. Please use the following commands to reproduce the results in the paper.

### Rotation on Mario-Iggy


Change to the mixmo folder:
```sh
cd examples/mixmo-pytorch; mkdir model_rotation;
```

Train the model:
```sh
python -m scripts.train --config_path config/marioiggy/exp_tinyimagenet_res18_1net_standard_bar1_test.yaml --dataplace $DATAPLACE --saveplace model_rotation --gpu 0 --info rotation --from_scratch --Li_config_path ../../InstaAug_module/configs/config_rotation_supervised.yaml
```

### Cropping on Tiny-Imagenet
Change to the mixmo folder:
```sh
cd examples/mixmo-pytorch; mkdir model_crop;
```

First per-train the model with no augmentation: 
```sh
python -m scripts.train --config_path config/tiny/exp_tinyimagenet_res18_1net_standard_bar1_test_pretrain.yaml --dataplace $DATAPLACE --saveplace model_crop/ --gpu 0 --info memory --from_scratch --max_tolerance 10
```

Then train with InstaAug:
```sh
python -m scripts.train --config_path config/tiny/exp_tinyimagenet_res18_1net_standard_bar1_test.yaml --dataplace $DATAPLACE --saveplace model_crop/ --gpu 0 --Li_config_path ../../InstaAug_module/configs/config_crop_supervised.yaml --max_tolerance 30 --max_no_decrease 50 --checkpoint model_crop/exp_tinyimagenet_res18_1net_standard_bar1_test_pretrainmemory/checkpoint_epoch_010.ckpt --resume_classifier_only --info crop --seed 3448
```

### Color-jittering on RawFooT
Change to the mixmo folder:
```sh
cd examples/mixmo-pytorch; mkdir model_color;
```

Train the model:
```sh
python -m scripts.train --config_path config/rawfoot/exp_tinyimagenet_res18_1net_standard_bar1_test_pretrain.yaml --dataplace $DATAPLACE --saveplace model_color/ --gpu 0 --info color --from_scratch --Li_config_path ../../InstaAug_module/configs/config_color_jittering_supervised.yaml
```

### Contrastive learning on Tiny-Imagenet
Change to the self-supervised folder:
```sh
cd examples/self-supervised; cp -r $DATAPLACE/tinyimagenet200 datasets;
```

Train the model:
```sh
python -m train --dataset tiny_in --epoch 500 --lr 2e-3 --emb 128 --method contrastive --model_folder model/test --Li_config_path ../../InstaAug_module/configs/config_crop_contrastive.yaml --eval_every 50 --crop_s0 1.0 --crop_s1 1.0 --crop_r0 1.0 --crop_r1 1.0 --wandb_name crop --entropy_weights 0.003 --num_workers 4 --target_entropy 3.7
```

## Data source 
* Mario-Iggy ([Download](https://github.com/g-benton/learning-invariances/tree/master/experiments/mario-iggy))
* Tiny-Imagenet ([Download](http://cs231n.stanford.edu/tiny-imagenet-200.zip))
* RawfooT ([Download](http://projects.ivl.disco.unimib.it/minisites/rawfoot/))  

Please cite original papers when using these datasets.

## References
If you find this code useful, please cite our paper.
```
@article{miao2022learning,
  title={Learning Instance-Specific Data Augmentations},
  author={Miao, Ning and Mathieu, Emile and Dubois, Yann and Rainforth, Tom and Teh, Yee Whye and Foster, Adam and Kim, Hyunjik},
  journal={arXiv preprint arXiv:2206.00051},
  year={2022}
}
```
