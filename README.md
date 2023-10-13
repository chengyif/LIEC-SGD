## Environment
Python 3.8 
PyTorch 1.10.1

## Parameters
- `--arch` name of model, which can be set to "resnet18" and "resnet34".
- `--dataset` name of dataset, which can be set to "cifar10" and "cifar100".
- `--method` name of algorithm, which can be set to "psgd", "memsgd", "doublesqueeze", "cser", "neolithic" and "liec".
- `--gpus` number of gpus used.
- `--lr` initial learning rate
- `--bucket-size` denote the size of the parameters in one block when adopting Blockwise-SignSGD, used to control the number of blocks. 
                  In the CIFAR-10 experiment, we set this parameter to 5242880(5*1024*1024) for Blockwise-SignSGD (10 blocks), 52428800 for SignSGD (1 block).
                  In the CIFAR-100 experiment, we set this parameter to 5242880(5*1024*1024) for Blockwise-SignSGD (18 blocks), 524288000 for SignSGD (1 block).   
- `--average-period` model average period of CSER and LIEC-SGD. In the CIFAR experiment, this parameter is set to 8 in CSER and 32 and 100 in LIEC-SGD.

## Example of experiments
- If train ResNet18 on CIFAR-10 dataset with P-SGD：
python main.py --arch resnet18 --dataset cifar10 --method psgd --batch-size 128 --gpus 8 --epochs 120 --lr 0.1
- If train ResNet18 on CIFAR-10 dataset with LIEC-SGD(32) using SignSGD：
python main.py --arch resnet18 --dataset cifar10 --method liec --batch-size 128 --gpus 8 --epochs 120 --lr 0.1 --bucket-size 52428800 --average-period 32
- If train ResNet34 on CIFAR-100 dataset with LIEC-SGD(100) using Blockwise-SignSGD：
python main.py --arch resnet34 --dataset cifar100 --method liec --batch-size 128 --gpus 8 --epochs 120 --lr 0.1 --bucket-size 5242880 --average-period 100

## Tiny-Imagenet experiments
The experiments on tiny-imagenet dataset is also run with the same code and parameters. The only change is that we adopt the cosine lr scheduler.