import argparse

def get_args_parser(parser):
    parser.add_argument('--model_name', default = "vit_B_16")
    parser.add_argument('--num_class', default = 10)
    parser.add_argument('--batch_size', default = 128)
    parser.add_argument('--img_size', default = 224)
    parser.add_argument('--patch_size', default = 16)
    parser.add_argument('--in_channels', default = 3)
    parser.add_argument('--depth', default = 12)
    parser.add_argument('--num_head', default = 12)
    parser.add_argument('--datasets', default = "CIFAR10")
    parser.add_argument('--data_path', default = "/data")
    parser.add_argument('--summary_path', default = "/summary")
    parser.add_argument('--lr', default = 0.001)
    parser.add_argument('--lr_momentum', default = 0.9)
    parser.add_argument('--weight_decay', default = 5e-4)
    parser.add_argument('--start_epoch', default = 0)
    parser.add_argument('--epochs', default = 100)


    