import argparse
import os
import wandb
import glob
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from model import NeuralNetwork
from dataset import PokemonDataset
from utils import EarlyStopper


config = {
    "lr": 1e-3,
    "dataset": "Pokemons",
    "epochs": 100,
    "batch_size": 256,
    "fc_layers": [64*16*16, 256], 
    "activations": "ReLU",
    "loss": "cross-entropy",
    "optimizer": "Adam",
    "augment": True
}

def calculate_acc(y_pred, y):
    preds = torch.argmax(y_pred, dim=1)
    num_correct = (preds == y).sum().item()

    return num_correct / len(y)


def train(device, dataloader, model, loss_fn, optimizer):
    model.train()
    avg_loss, avg_acc = 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        # Get prediction and compute loss
        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        avg_loss += loss.item()

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_acc += calculate_acc(y_pred, y)
    
    avg_loss /= len(dataloader)
    avg_acc /= len(dataloader)
        
    return (avg_loss, avg_acc)


def val(device, dataloader, model, loss_fn):    
    model.eval()
    avg_loss, avg_acc = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            avg_loss += loss_fn(y_pred, y).item()
            avg_acc += calculate_acc(y_pred, y)
    
    avg_loss /= len(dataloader)
    avg_acc /= len(dataloader)
    
    return (avg_loss, avg_acc)


def prepare_data(data_dir):
    labels = os.listdir(data_dir)
    data = glob.glob(os.path.join(data_dir, '*/*.jpg'))
    # print(len(data))
    # print(data[:5])
    # print(labels[len(labels)-10:])

    labels_map = dict()
    reversed_labels_map = dict()
    for i, label in enumerate(labels):
        labels_map[i] = label
        reversed_labels_map[label] = i
    # print(labels_map)
    # print(reversed_labels_map)

    data_train, data_val = train_test_split(data, test_size=0.3, train_size=0.7, random_state=420)
    print(f'train: {len(data_train)}, val: {len(data_val)} | {len(data_train)+len(data_val)}')

    train_data = PokemonDataset(data_train, data_dir, reversed_labels_map, transform=transforms.Resize((128, 128)), augment=config['augment'])
    val_data = PokemonDataset(data_val, data_dir, reversed_labels_map, transform=transforms.Resize((128, 128)), augment=False)

    train_dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)

    return train_dataloader, val_dataloader, labels_map


# def test_dataloader(dataloader, labels_map, name):
#     fig, axs = plt.subplots(2, 5, figsize=(10, 5))
#     axs = axs.flatten()

#     images, labels = next(iter(dataloader))
#     for i, (img, label) in enumerate(zip(images, labels)):
#         if i == 10: break
#         axs[i].imshow(img[:,:,:].permute(1, 2, 0))
#         axs[i].set_title(f'{label.item()}, {labels_map[label.item()]}', fontsize=9)
#         axs[i].axis('off')
#     fig.savefig(f'outputs/{name}')
#     plt.close(fig)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to data')
    parser.add_argument('--wandb', type=str, help='wandb id')
    args = parser.parse_args()

    use_wandb = False
    wandb_key = args.wandb
    if wandb_key:
        wandb.login(key=wandb_key)
        wandb.init(project="assignment-2", name="", reinit=True, config=config)    
        use_wandb = True

    if args.data_path:
        data_dir = args.data_path
    else:
        print('You didnt specify the data path >:(')
        return
    
    # Training 
    train_dataloader, val_dataloader, labels_map = prepare_data(data_dir)

    if not os.path.isdir('outputs'):
        os.mkdir('outputs')

    # test_dataloader(train_dataloader, labels_map, 'train')
    # test_dataloader(val_dataloader, labels_map, 'val')

    model = NeuralNetwork().to(device)
    print(model)

    early_stopper = EarlyStopper(patience=5, min_delta=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = nn.CrossEntropyLoss()

    epochs = config['epochs']
    train_history = {'loss': [], 'acc': []}
    val_history = {'loss': [], 'acc': []}        
    for epoch in range(epochs):
        train_loss, train_acc = train(device, train_dataloader, model, loss_fn, optimizer)
        val_loss, val_acc = val(device, val_dataloader, model, loss_fn)

        train_history['loss'].append(train_loss)
        train_history['acc'].append(train_acc)

        val_history['loss'].append(val_loss)
        val_history['acc'].append(val_acc)
        
        # if (epoch % 10 == 0):
        print(f'Epoch {epoch}')
        print(f'loss: {train_loss:>5f} acc: {train_acc:>5f}')
        print(f'val loss: {val_loss:>5f} val acc: {val_acc:>5f}')
        # print(f'lr: {optimizer.param_groups[0]["lr"]:>5f}')
        print('-------------------------------')

        if use_wandb:
            wandb.log({
                'epoch': epoch, 'loss': train_loss, 'accuracy': train_acc, 
                'val_loss':val_loss, 'val_accuracy': val_acc, 'lr': optimizer.param_groups[0]["lr"]
            })

        if early_stopper(val_loss):
            print('Stopping early!!!')
            break
    
    
    torch.save(model, 'outputs/model.pt')
    if use_wandb:
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file('outputs/model.pt')
        wandb.log_artifact(artifact)
        wandb.finish()


if __name__ == '__main__':
    main()