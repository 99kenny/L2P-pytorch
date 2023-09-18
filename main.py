from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
import config
import torch
from data import data_util
from layers import vision_transformer
from data import data_util
import torch.nn as nn
import utils

configs = config.Config()
model = vision_transformer.VisionTransformer(configs.num_head, configs.num_class, configs.batch_size)
model.cuda()
train_transforms = data_util.get_transforms(is_train=True)
val_transforms = data_util.get_transforms(is_train=False)

train_datasets, val_datasets = data_util.get_dataset("notMNIST", train_transforms, val_transforms, data_path=configs.data_path, download=False)

train_loader = DataLoader(
    train_datasets,
    batch_size = configs.batch_size,
    shuffle = True,
)

val_loader = torch.utils.data.DataLoader(
    train_datasets,
    batch_size = configs.batch_size,
    shuffle = True,
)

writer = SummaryWriter()

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), configs.lr, weight_decay=configs.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

if not torch.cuda.is_available():
    print("cuda not enabled")
    raise ValueError

for epoch in range(configs.start_epoch, configs.epochs):
    print("current lr {:.5e}".format(optimizer.param_groups[0]['lr']))
    utils.train(train_loader, model, criterion, optimizer, epoch, writer)
    lr_scheduler.step()
    avg_acc = utils.validate(val_loader, model, criterion, epoch)
    
    is_best = avg_acc > best_acc
    if is_best:
        torch.save(model.state_dict(), "{}/best.pth".format(configs.summary_path))
        best_acc = avg_acc    