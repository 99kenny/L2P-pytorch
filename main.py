import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
import torch.nn as nn

from config import *
from data import *
from utils import *
from vision_transformer import *

import argparse

def main(args):
    configs = args
    model = VisionTransformer(configs.num_head, configs.num_class, configs.batch_size)
    
    # load weights to the model
    download_pretrained()
    
    # test 
    model.cuda()

    train_transforms = get_transforms(is_train=True)
    val_transforms = get_transforms(is_train=False)

    train_datasets, val_datasets = get_dataset(configs.datasets, train_transforms, val_transforms, data_path=configs.data_path, download=True)

    train_loader = DataLoader(
        train_datasets,
        batch_size = configs.batch_size,
        shuffle = True,
    )

    val_loader = DataLoader(
        val_datasets,
        batch_size = configs.batch_size,
        shuffle = True,
    )

    writer = SummaryWriter(configs.summary_path)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), configs.lr, weight_decay=configs.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

    if not torch.cuda.is_available():
        print("cuda not enabled")
        raise ValueError

    for epoch in range(configs.start_epoch, configs.epochs):
        print("current lr {:.5e}".format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch, writer)
        lr_scheduler.step()
        avg_acc = validate(val_loader, model, criterion, epoch, writer)
        
        is_best = avg_acc > best_acc
        if is_best:
            torch.save(model.state_dict(), "{}/best.pth".format(configs.summary_path))
            best_acc = avg_acc    
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser("ViT")
    
    # parser 
    args = parser.parse_args()
    get_args_parser(args)
    main(args)