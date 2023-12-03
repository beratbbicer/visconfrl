import random
fixed_seed = random.randint(0, 2**32 - 1)
random.seed(fixed_seed)
import numpy as np
np.random.seed(fixed_seed)
import torch
torch.manual_seed(fixed_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(fixed_seed)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import pickle
import argparse
import string
from pathlib import Path
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def initialize_bias_conv(layer):
    if layer.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(layer.bias, -bound, bound)

def initialize_bias_linear(layer):
    if layer.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(layer.bias, -bound, bound)

def conv_output_shape(x, kernel_size=1, stride=1, pad=0, dilation=1):
    return math.floor(((x + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)

class SkipLayer(nn.Module):
    def __init__(self, other):
        super(SkipLayer, self).__init__()
        self.other = other
    
    def forward(self, x):
        out = self.other(x)
        return out + x

class BaseModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, numlayers):
        super(BaseModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.numlayers = numlayers

        # ====================================================================================
        # FE Layers
        # ====================================================================================
        cur_dim = min(self.state_dim[0], self.state_dim[1])
        assert cur_dim >= 3, "State dimensions must be at least 3x3"
        layers, channels = [], 1

        for i in range(numlayers):
            conv = nn.Conv2d(channels, hidden_dim, kernel_size=3, stride=1, padding=1, padding_mode="replicate")
            torch.nn.init.kaiming_uniform_(conv.weight, mode="fan_out", nonlinearity="leaky_relu")
            initialize_bias_conv(conv)
            layers += [SkipLayer(conv), nn.LeakyReLU(negative_slope=0.01), nn.BatchNorm2d(hidden_dim)]
            channels = hidden_dim

        while cur_dim >= 2:
            conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, padding_mode="replicate")
            torch.nn.init.kaiming_uniform_(conv.weight, mode="fan_out", nonlinearity="leaky_relu")
            initialize_bias_conv(conv)
            layers += [conv, nn.LeakyReLU(negative_slope=0.01), nn.BatchNorm2d(self.hidden_dim)]
            cur_dim = conv_output_shape(cur_dim, kernel_size=3, stride=2, pad=1)
        self.layers = nn.ModuleList(layers)

        # ====================================================================================
        # Output Layer
        # ====================================================================================
        layer = nn.Linear(hidden_dim, action_dim)
        torch.nn.init.kaiming_uniform_(layer.weight, mode="fan_out", nonlinearity="linear")
        initialize_bias_linear(layer)
        self.output_layer = layer

    def forward(self, state):
        if len(state.size()) == 2:
            b,h,w = state.unsqueeze(0).size()
        elif len(state.size()) == 3:
            b,h,w = state.size()
        else:
            b,_,h,w = state.size()
        out = state.view(b, 1, h, w)

        for layer in self.layers:
            out = layer(out)
        out = F.adaptive_avg_pool2d(out, output_size=(1,1)).view(b, -1)
        out = F.softmax(self.output_layer(out), dim=-1)
        return out

class GridViewDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def split_data(path, savepath=None):
    with open(path, "rb") as f:
        data = list(pickle.load(f)['views'].values())
        
    random.shuffle(data)
    train_data = data[:int(len(data)*0.8)]
    val_data = data[int(len(data)*0.8):int(len(data)*0.9)]
    test_data = data[int(len(data)*0.9):]
    if savepath:
        with open(savepath, "wb") as f:
            pickle.dump([train_data, val_data, test_data], f)
    return GridViewDataset(train_data), GridViewDataset(val_data), GridViewDataset(test_data)

def get_datasets(path):
    with open(path, "rb") as f:
        train_data, val_data, test_data = pickle.load(f)
    return GridViewDataset(train_data), GridViewDataset(val_data), GridViewDataset(test_data)

def collate_fn(batch):
    views, rewards = zip(*batch)
    views = torch.from_numpy(np.stack(views, axis=0)).float()
    rewards = torch.from_numpy(np.stack(rewards, axis=0)).float()
    return views, rewards

def load_model(path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model = BaseModel(checkpoint['args'].state_dim, checkpoint['args'].action_dim, checkpoint['args'].hidden_dim, checkpoint['args'].numlayers)
    # if checkpoint['args'].double:
    #     model = model.double()
    # model = model.to(checkpoint['args'].device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint

def run_experiment(args, model, train_dl, val_dl, test_dl, optimizer, scheduler, criterion, training_file, test_file):
    epoch_train_losses, epoch_val_losses, epoch_test_losses = [], [], []
    for epoch in range(args.max_epochs):
        # ==========================================================================
        # Training
        # ==========================================================================
        _ = torch.set_grad_enabled(True)
        model.train()
        train_losses = []

        for train_minibatch, (views, rewards) in enumerate(train_dl):
            optimizer.zero_grad()
            if args.double:
                views = views.double()
                rewards = rewards.double()
            views = views.to(args.device)
            rewards = rewards.to(args.device)               
            out = model(views)
            loss = criterion(out, rewards)
            train_losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gcnorm)
            optimizer.step()
            # break
        
        train_loss = np.mean(train_losses)
        epoch_train_losses.append(train_loss)
        with np.printoptions(precision=4, suppress=True, threshold=5):
            print(f"Epoch {epoch:3d}/{args.max_epochs-1:3d} -> train_loss: {train_loss:.4f}")
            training_file.write(f"Epoch {epoch:3d}/{args.max_epochs-1:3d} -> train_loss: {train_loss:.4f}\n")
        
        # ==========================================================================
        # Validation
        # ==========================================================================
        model.eval()
        val_losses = []

        for val_minibatch, (views, rewards) in enumerate(val_dl):
            if args.double:
                views = views.double()
                rewards = rewards.double()
            views = views.to(args.device)
            rewards = rewards.to(args.device)               
            out = model(views)
            loss = criterion(out, rewards)
            val_losses.append(loss.item())
            # break

        val_loss = np.mean(val_losses)
        epoch_val_losses.append(val_loss)
        scheduler.step()
        with np.printoptions(precision=4, suppress=True, threshold=5):
            print(f"Epoch {epoch:3d}/{args.max_epochs-1:3d} ->   val_loss: {val_loss:.4f}")
            training_file.write(f"Epoch {epoch:3d}/{args.max_epochs-1:3d} ->   val_loss: {val_loss:.4f}\n")

        # ==========================================================================
        # Testing
        # ==========================================================================
        test_losses = []

        for test_minibatch, (views, rewards) in enumerate(test_dl):
            if args.double:
                views = views.double()
                rewards = rewards.double()
            views = views.to(args.device)
            rewards = rewards.to(args.device)               
            out = model(views)
            loss = criterion(out, rewards)
            test_losses.append(loss.item())
            # break
        
        test_loss = np.mean(test_losses)
        epoch_test_losses.append(test_loss)
        with np.printoptions(precision=4, suppress=True, threshold=5):
            test_file.write(f"Epoch {epoch:3d}/{args.max_epochs-1:3d} -> test_loss: {test_loss:.4f}\n")

        if args.dump:
            torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "criterion": criterion.state_dict(),
                "fixed_seed ": fixed_seed,
                "args": args,
            }, f"{args.writepath}/e{epoch}.pth")

    fig = plt.figure(figsize=(10, 4))
    x = np.arange(len(epoch_train_losses))
    plt.xticks(ticks=x, labels=x, rotation=90)
    plt.plot(x, epoch_train_losses, label="Train Loss", color="red", linewidth=2, linestyle=":", marker="o")
    plt.plot(x, epoch_val_losses, label="Val Loss", color="blue", linewidth=2, linestyle=":", marker="o")    
    plt.plot(x, epoch_test_losses, label="Test Loss", color="green", linewidth=2, linestyle=":", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Throughout Training")
    plt.legend()
    plt.tick_params(axis="x", which="major", labelsize=5)
    plt.tick_params(axis="y", which="major", labelsize=10)
    plt.tight_layout()
    plt.savefig(f"{args.writepath}/figure.png")

    with open(f"{args.writepath}/figure.pkl", "wb") as f:
        pickle.dump({
                "fig": fig,
                "epoch_train_losses": epoch_train_losses,
                "epoch_val_losses": epoch_val_losses,
                "epoch_test_losses": epoch_test_losses,
            }, f)

def initialize_experiment(args, settings):
    model = BaseModel(args.state_dim, args.action_dim, args.hidden_dim, args.numlayers).to(args.device)
    if args.double:
        model = model.double()

    train_ds, val_ds, test_ds = get_datasets(args.gridpath)
    train_dl, val_dl, test_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn),\
                                DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn),\
                                DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.max_epochs*0.25),int(args.max_epochs*0.5),int(args.max_epochs*0.75)], gamma=0.5)
    criterion = nn.L1Loss()
    training_file = open(f'{args.writepath}/training.txt', 'w')
    training_file.write(f'{settings}\n')
    test_file = open(f'{args.writepath}/test.txt', 'w')
    run_experiment(args, model, train_dl, val_dl, test_dl, optimizer, scheduler, criterion, training_file, test_file)
    training_file.close()
    test_file.close()

def initialize_pretrain():
    parser = argparse.ArgumentParser(description='RL Agent pretraining.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--gridpath', default='./grid_views/2_101_101_0.9_0.9_13000_600_split.pkl', type=str, help='Path to grid views .pkl archive.')
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--numlayers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight_decay", default=2e-5, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument('--max_epochs', default=100, type=int, help="Number of epochs. Default is 500.")
    parser.add_argument('--gcnorm', default=2.0, type=int, help="Max norm for gradient clipping. Default is 2.0.")
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--rootname", type=str, help="Name for experiment root folder. Defaults to length-8 random string.",\
                        default="".join(random.SystemRandom().choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(8)))
    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.fixed_seed = fixed_seed
    args.view_kernel_size = int(args.gridpath.split("/")[-1].split(".pkl")[0].split("_")[0])
    args.state_dim, args.action_dim = (args.view_kernel_size*2+1, args.view_kernel_size*2+1), 4

    settings = f"Configuration pretrain_{args.rootname} ->\n\
        hidden_dim:{args.hidden_dim}, hidden_dim:{args.hidden_dim}, numlayers:{args.numlayers}, lr:{args.lr:.5f}, weight_decay:{args.weight_decay:.5f}\n\
        batch_size:{args.batch_size}, max_epochs:{args.max_epochs}, gcnorm:{args.gcnorm}, device:{args.device}, double:{args.double}, dump:{args.dump}\n"
    
    writepath = f'./experiments/pretrain/{args.gridpath.split("/")[-1].split(".pkl")[0]}/{args.rootname}'
    Path(writepath).mkdir(parents=True, exist_ok=True)
    args.writepath = writepath
    print(settings)
    args.dump = True
    args.double = True
    initialize_experiment(args, settings)

if __name__ == "__main__":
    # =============================================================
    '''
    w,h,action_dim,hidden_dim,numlayers = 5,5,4,256,8
    model = BaseModel((h,w), action_dim, hidden_dim, numlayers)
    state = torch.randn(100, h, w)
    out = model(state)
    print(out.size())
    '''
    # =============================================================
    '''
    train_ds, val_ds, test_ds = split_data('./grid_views/2_101_101_0.9_0.9_13000_600.pkl', './grid_views/2_101_101_0.9_0.9_13000_600_split.pkl')
    train_dl, val_dl, test_dl = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn),\
                                DataLoader(val_ds, batch_size=256, shuffle=True, collate_fn=collate_fn),\
                                DataLoader(test_ds, batch_size=256, shuffle=True, collate_fn=collate_fn)
    
    for i, (views, rewards) in enumerate(train_dl):
        print(views.size())
        break
    '''
    # =============================================================
    initialize_pretrain()