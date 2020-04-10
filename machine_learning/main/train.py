import __init_lib_path
from config_guard import cfg, update_config_from_yaml
import dataset
import network
import classifier
import loss

import argparse
import torch
import torch.nn as nn
import torch.optim as optim

def parse_args():
    parser = argparse.ArgumentParser(description = "Roger's Deep Learning Playground")
    parser.add_argument('--cfg', help = "specify particular yaml configuration to use", required = True,
        default = "configs/mnist_torch_official.taml", type = str)
    args = parser.parse_args()

    return args

def train(cfg, model, classifier, criterion, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if cfg.BACKBONE.forward_need_label:
            # For certain task (e.g. CVAE), the network takes label as inputs
            logits = model(data, target)
        else:
            logits = model(data)
        output = classifier(logits)
        if cfg.task == "auto_encoder":
            output = output.reshape(data.shape)
            loss = criterion(output, data, model.aux_dict)
        else:
            loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % cfg.TRAIN.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(cfg, model, classifier, criterion, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if cfg.BACKBONE.forward_need_label:
                # For certain task (e.g. CVAE), the network takes label as inputs
                logits = model(data, target)
            else:
                logits = model(data)
            output = classifier(logits)
            if cfg.task == "auto_encoder":
                output = output.reshape(data.shape)
                test_loss += criterion(output, data, model.aux_dict).item()
            else:
                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim = 1, keepdim = True)
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if cfg.task == "auto_encoder":
        print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    else:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def main():
    # --------------------------
    # | Initial set up
    # |  1. Parse arg and yaml
    # |  2. Set device
    # |  3. Set seed
    # --------------------------
    args = parse_args()
    update_config_from_yaml(cfg, args)

    use_cuda = not cfg.SYSTEM.use_cpu
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': cfg.SYSTEM.num_workers, 'pin_memory': cfg.SYSTEM.pin_memory} if use_cuda else {}

    torch.manual_seed(cfg.seed)

    # --------------------------
    # | Prepare datasets
    # --------------------------
    train_set, test_set = dataset.dispatcher(cfg)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.TRAIN.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.TEST.batch_size, shuffle=True, **kwargs)

    # --------------------------
    # | Get ready to learn
    # |  1. Prepare network and loss
    # |  2. Prepare optimizer
    # |  3. Set learning rate
    # --------------------------
    Net = network.dispatcher(cfg)
    model = Net(cfg).to(device)

    logit_processor = classifier.dispatcher(cfg)

    criterion = loss.dispatcher(cfg)

    optimizer = optim.Adadelta(model.parameters(), lr = cfg.TRAIN.initial_lr)

    # Prepare LR scheduler
    if cfg.TRAIN.lr_scheduler == "step_down":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.TRAIN.step_down_on_epoch, cfg.TRAIN.step_down_gamma)

    for epoch in range(1, cfg.TRAIN.max_epochs + 1):
        train(cfg, model, logit_processor, criterion, device, train_loader, optimizer, epoch)
        test(cfg, model, logit_processor, criterion, device, test_loader)
        scheduler.step()

    if cfg.save_model:
        torch.save(model.state_dict(), "{0}_final.pt".format(cfg.name))


if __name__ == '__main__':
    main()