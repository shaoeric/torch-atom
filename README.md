# Torch-atom

[TOC]



## Introduction

A basic and simple training framework for pytorch, easy for extension.

**architecture figure**

## Dependence

- torch==1.7.0+cu110
- torchvision==0.8.0
- easydict==1.9
- tensorboard==2.7
- tensorboardX==2.4
- PyYAML==5.3.1

## Features

- Any module can be easily customized
- Not abstract, easy to learn, develop and debug
- With a lot of repetitive work reduction, it can be more easily control the training process in development and research
- Friendly to multi-network interactive training, such as GAN, transfer learning, knowledge distillation, etc.

## Train

```shell
python main.py --model resnet18 --save_dir cifar100_resnet18
```



## Results

### CIFAR100


|   Network    | Accuracy | log&ckpt |
| :----------: | :------: | :------: |
|   resnet18   |  76.46   |          |
|   resnet34   |  77.23   |          |
|   resnet50   |  76.82   |          |
|  resnet101   |  77.32   |          |
|   vgg11_bn   |  70.52   |          |
|   vgg13_bn   |  73.71   |          |
| mobilenetV2  |  68.99   |          |
|  shufflenet  |  71.17   |          |
| shufflenetV2 |  71.16   |          |



## Customize

### Customize Dataset



### Customize Model

- In `src/models` directory, define your customized model, such as `my_model.py` , and define the module class `MyModel`. Please refer to `resnet.py`
- In `src/models/model_builder.py`, import your model. `from .my_model import *` under the `try` process, and `from my_model import *` under the `except` process. It is just convenient for debugging.
- In `configs/xxx.yml`, set the `model['name']` to `MyModel` 

### Customize Loss Function



### Customize Optimizer

- In `src/optimizer` directory, `optimizers.py` can be found, please define your customized optimizer here. For example, `SGD` and `Adam` have already defined, `parameters` and `lr` should be specified, and other params need to be specifed by `*args, **kwargs`. Please refer to

  ```python
  # src/optimizer/optimizers.py
  def SGD(parameters, lr, *args, **kwargs) -> optim.Optimizer:
      optimizer = optim.SGD(parameters, lr=lr, *args, **kwargs)
      return optimizer
  ```

- For other parameters, such as `weight_decay`, can be set in `src/optimizer/optimizer_config.yml`. Please refer to the below yaml, and it is ok for `5e-4` format, we transform it in `src/optimizer/optimizer_builder.py`. 

  ```yaml
  # src/optimizer/optimizer_config.yml
  SGD:
    momentum: 0.9
    weight_decay: 5e-4
    dampening: 0
    nesterov: False
  ```
  
- In `configs/xxx.yml`, set the `train['lr']`, and set the `train['optimizer']` to `SGD`



### Customize Schemes

-  



### Customize Metrics





### Customize Training and Evaluation Procedure for One Batch





### Customize Checkpoint Saving Strategy





## todo

- Rewrite data augmentation code
- DDP training
- More experiment results
- Some visualization code for analysis
  - bad case analysis
  - data augmentation visualization
  - ...



## Acknowledgement

Torch-atom got ideas and developed based on the following projects:

[open-mmlab/mmclassification](https://github.com/open-mmlab/mmclassification)

[weiaicunzai/pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100)



## Citation

If you find this project useful in your research, please consider cite:

```
@misc{2022torchatom,
    title={Torch-atom: A basic and simple training framework for pytorch},
    author={Baitan Shao},
    howpublished = {\url{https://github.com/shaoeric/torch-atom}},
    year={2022}
}
```

## License

[The MIT License | Open Source Initiative](https://opensource.org/licenses/MIT)