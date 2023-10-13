import argparse

def GetArgs():
    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('--path', default='/data/', type=str, help='path to dataset')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'tiny-imagenet'])
    parser.add_argument('-a', '--arch', default='resnet18', help='model architecture')
    parser.add_argument('-j', '--workers', default=4, type=int, help='data loading workers')
    parser.add_argument('--epochs', default=120, type=int, help='total epochs')
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('-eb', '--eval-batch-size', default=128, type=int)
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--gpus', default=4, type=int, help='number of gpus per node')
    parser.add_argument('--process_num',type=int,default=1, help='the number of process per gpu')
    parser.add_argument('--rank', default=0, type=int, help='node rank')
    parser.add_argument('--root', default=0, type=int, help='root node')
    parser.add_argument('--st', type=int, default=0, help='gpu start')
    parser.add_argument('--seed', default=0, type=int, help='dist sampler')
    parser.add_argument('--dist-url', default='tcp://localhost:23450', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='GLOO', type=str, help='dist backend')
    parser.add_argument('--method', default='psgd', type=str)

    ### optimizer hyperparameters ###
    parser.add_argument('-opt', '--optimizer', type=str, default='sgd', help='choose from (sgd, adagrad)')
    parser.add_argument('-wd', '--weight-decay', default=5e-4, type=float)
    parser.add_argument('-m', '--momentum', default=0, type=float)
    parser.add_argument('-bs', '--bucket-size', default=52428800, type=int)
    parser.add_argument('-ap', '--average-period', default=100, type=int)


    ### learning rate ###
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('-ls', '--lr-schedule', default='cos', type=str, choices=['const', 'cos'])
    parser.add_argument('-ds', '--decay-schedule', type=int, nargs='+', default=[80,160],
                    help='learning rate decaying epochs')


    return parser.parse_args()
