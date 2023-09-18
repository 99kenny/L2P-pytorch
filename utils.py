import torch

def accuracy(output, y):
    pred = output.topk(k=1).values
    pred = torch.reshape(pred, (-1,))
    guess = pred - y
    acc = len((guess == 0).nonzero()) / len(y) 
    return acc
    
def train(train_loader, model, criterion, optimizer, epoch, writer):
    losses = 0.
    accs = 0.
    # train mode
    model.train()
    # one epoch
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        
        #forward
        output = model(input_var)
        loss = criterion(output, target_var)
        
        # compute grad
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        output = output.float()
        losses += loss.float()
        accs += accuracy(output.data, target)
        
    losses /= len(train_loader)
    accs /= len(train_loader)
    writer.add_scalar("Loss/train", losses, epoch)
    writer.add_scalar("Accuracy/train", accs, epoch)
    
    return accs
    
def validate(val_loader, model, criterion, epoch, writer):
    losses = 0.
    accs = 0.

    # evaluation mode
    model.eval()
    
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            output = output.float()
            loss = loss.float()
            
            # measure accuracy and record loss
            accs += accuracy(output.data, target)
            losses += loss
            
        losses /= len(val_loader)
        accs /= len(val_loader)
        writer.add_scalar("Loss/val", losses, epoch)
        writer.add_scalar("Accuracy/val", accs, epoch)
        
    return accs