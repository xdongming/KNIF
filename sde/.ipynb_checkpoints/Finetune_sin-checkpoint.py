import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.func import vmap, jacrev
from tqdm import tqdm
import os
import random
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
import math
from pydmd import DMD
from sklearn.preprocessing import MinMaxScaler
import warnings
from pytorch_lightning.strategies import DDPStrategy
import copy

class ResidualFlow(nn.Module):
    def __init__(self, dim, hidden_dim, n_layers, input_dim=0, dropout=0, LDJ=False):
        super(ResidualFlow, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.LDJ = LDJ
        self.dropout = dropout
        
        layers = [nn.Linear(self.dim+self.input_dim, self.hidden_dim), nn.ReLU()]
        for _ in range(self.n_layers):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(self.hidden_dim, self.dim))
        self.net = nn.Sequential(*layers)
        self._initialize_weights()
    
    def forward(self, x, u=None, reverse=False):
        def func(x_):
            x_e = torch.cat((x_, u), dim=-1) if u is not None else x_
            return self.net(x_e)
        if not reverse:   
            y = x + func(x)
            if self.LDJ:
                x = x.view(-1, x.shape[-1])
                jacobian = vmap(jacrev(func))(x)
                jacobian = jacobian.clone()
                jacobian.diagonal(dim1=-2, dim2=-1).add_(1.0)
                _, logdet = torch.linalg.slogdet(jacobian)
                logdet = logdet.sum()
            else:
                logdet = 0
            return y, logdet
        else:
            y = x
            epsilon = 1e-4
            det = 1
            max_iter = 100
            with torch.no_grad():
                for _ in range(max_iter):
                    y_temp = y
                    y = x - func(y)
                    det = torch.norm(y - y_temp, dim=-1).max()
                    if det < epsilon:
                        break  
                # while det > epsilon:
                #     y_temp = y
                #     y = x - func(y)
                #     det = torch.norm(y - y_temp, dim=1).max()
            return y
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):  
                nn.init.xavier_uniform_(module.weight)  
                if module.bias is not None:
                    nn.init.zeros_(module.bias)  

class InvertibleNN(nn.Module):
    def __init__(self, dim, hidden_dim, n_blocks, n_layers, input_dim=0, dropout=0, LDJ=False):
        super(InvertibleNN, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.blocks = nn.ModuleList([ResidualFlow(self.dim, self.hidden_dim, self.n_layers, self.input_dim, dropout, LDJ) for _ in range(self.n_blocks)])
    
    def forward(self, x, u=None, reverse=False):
        if not reverse:
            ldj_total = 0
            for block in self.blocks:
                x, ldj = block(x, u, reverse)
                ldj_total += ldj
            return x, ldj_total
        else:
            for block in reversed(self.blocks):
                x = block(x, u, reverse)
            return x
    
class CombinedNetwork(nn.Module):
    def __init__(self, inn_model, input_dim, lifted_dim, Xmax, Xmin):
        super(CombinedNetwork, self).__init__()
        self.input_dim = input_dim
        self.inn_model = inn_model  
        self.Xmax = Xmax
        self.Xmin = Xmin
        self.dropout = nn.Dropout(p=inn_model.blocks[0].dropout)
        self.linear = nn.Linear(input_dim, lifted_dim, bias=False)  
        self._initialize_weights()
    
    def forward(self, x, u=None, reverse=False):
        x = x.float()
        Xmax = self.Xmax.to(x.device)
        Xmin = self.Xmin.to(x.device)
        if not reverse:
            x = (x - Xmin) / (Xmax - Xmin)
            chebyshev = torch.cos(self.linear(torch.arccos(x)))
            x = torch.cat((x, chebyshev), dim=-1)
            # x = self.dropout(x)
            x, ldj = self.inn_model(x, u, reverse)
            return x, ldj
        else:
            x = self.inn_model(x, u, reverse)
            x = x[:, :self.input_dim]
            x = (Xmax - Xmin) * x + Xmin
            return x
    
    def _initialize_weights(self):
        lambda_s = 5
        self.linear.weight.data = torch.distributions.exponential.Exponential(lambda_s).sample(self.linear.weight.shape)
        
class TrainModel(pl.LightningModule):
    def __init__(self, model, rank, learning_rate=1e-3, lamb=1, path="model_checkpoint_Van"):
        super(TrainModel, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        self.best_val_loss = float('inf')  
        self.validation_outputs = []
        self.lamb = lamb
        self.train_losses = []
        self.rank = rank
        self.path = path+'.ckpt'

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X_batch = batch[0]
        GY, GY_pred, ldj = dmd(self.model, X_batch, self.rank)

        loss_lin = self.criterion(GY, GY_pred)
        loss_LDJ = ldj / X_batch.numel()

        loss = loss_lin - self.lamb * loss_LDJ
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        Z_batch = batch[0]
        Z1, Z_pred, _ = dmd(self.model, Z_batch, self.rank)
        Z_pred = self.model(Z_pred.T, reverse=True)
        Z1 = self.model(Z1.T, reverse=True)
        valid_loss = self.criterion(Z_pred, Z1)

        self.validation_outputs.append(valid_loss)
        self.log('val_loss', valid_loss)
        return valid_loss

    def test_step(self, batch, batch_idx):
        Z_batch = batch[0]
        Z1, Z_pred, _ = dmd(self.model, Z_batch, self.rank)
        Z_pred = self.model(Z_pred.T, reverse=True)
        Z1 = self.model(Z1.T, reverse=True)
        test_loss = self.criterion(Z_pred, Z1)

        self.log('test_loss', test_loss)
        return test_loss
    
    def on_fit_start(self):
        if self.trainer.is_global_zero: 
            if os.path.exists("loss_log.txt"):
                os.remove("loss_log.txt")
            if os.path.exists(self.path):
                os.remove(self.path)
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.no_grad():  
            for name, module in self.model.named_modules():  
                if isinstance(module, nn.Linear): 
                    if name == "linear":  
                        continue
                    weight = module.weight  
                    sigma_max = torch.norm(weight, p=2)  
                    if sigma_max > 1:  
                        scale = (1 - 1e-3) / sigma_max
                        module.weight.data *= scale  
    
    def on_train_epoch_start(self):
        if os.path.exists(self.path):
            best_state_dict = torch.load(self.path)["state_dict"]
            self.load_state_dict(best_state_dict)
    
    def on_train_epoch_end(self):
        if self.trainer.is_global_zero:  
            avg_train_loss = self.trainer.callback_metrics.get("train_loss")
            if avg_train_loss is not None:
                self.train_losses.append(avg_train_loss.item()) 
                print(f"Epoch {self.current_epoch}: Average Training Loss = {avg_train_loss.item()}")

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.validation_outputs).mean() 
        self.log('avg_val_loss', avg_val_loss)
        self.validation_outputs.clear()
        print(f"Validation loss: {avg_val_loss}")
        with open("loss_log.txt", "a") as f:
            f.write(f"{avg_val_loss.item()}\n")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-08,
                                            weight_decay=0)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='min',
        #     factor=0.5,
        #     patience=5,
        #     cooldown=1
        # )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.92
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  
            },
            "gradient_clip_val": 1.0,  
            "gradient_clip_algorithm": "norm",
        }
        
def load_g_model(checkpoint_path, model):
    lightning_model = TrainModel.load_from_checkpoint(checkpoint_path, model=model, rank=rank, learning_rate=learning_rate, map_location="cpu")
    return lightning_model.model.eval()

def save_model_and_K(g_model, K, save_dir="checkpoints", model_name="g_model"):
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save(g_model.state_dict(), model_path)

    K_path = os.path.join(save_dir, f"{model_name}_K.pt")
    torch.save(K, K_path)

    print(f"Saved g_model to {model_path}")
    print(f"Saved Koopman operator K to {K_path}")

def load_model_and_K(model_class, K_path, model_path, device="cpu", **model_kwargs):
    g_model = model_class(**model_kwargs).to(device)
    g_model.load_state_dict(torch.load(model_path, map_location=device))
    g_model.eval()

    K = torch.load(K_path, map_location=device)

    print(f"Loaded g_model from {model_path}")
    print(f"Loaded Koopman operator K from {K_path}")

    return g_model, K

def spectral_radius(matrix):
    eigvals = torch.linalg.eigvals(matrix)
    return torch.max(torch.abs(eigvals)).item()

def spectral_norm_g_model(g_model, exclude_name="linear", max_radius=1.0):
    with torch.no_grad():
        for name, module in g_model.named_modules():
            if isinstance(module, nn.Linear) and name != exclude_name:
                radius = torch.linalg.norm(module.weight, ord=2)
                if radius >= max_radius:
                    module.weight.data *= (max_radius - 1e-3) / radius

def train_alternating(
    g_model, H_train, dim, lifted_dim,
    num_epochs=50, batch_size=64, lr_g=1e-3, lr_K=1e-2, lamb=1e-3,
    save_dir="checkpoints", model_name="g_model"
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g_model.to(device)

    x = torch.tensor(H_train[:, :-1, :], dtype=torch.float32).to(device)
    y = torch.tensor(H_train[:, 1:, :], dtype=torch.float32).to(device)

    x_all = x.reshape(-1, x.shape[-1])
    y_all = y.reshape(-1, y.shape[-1])
    dataset = TensorDataset(x_all, y_all)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    with torch.no_grad():
        gx_all = g_model(x_all)[0]
        gy_all = g_model(y_all)[0]
        K_param = nn.Parameter(torch.linalg.lstsq(gx_all, gy_all).solution.detach())

    optimizer_g = optim.AdamW(g_model.parameters(), lr=lr_g, weight_decay=1e-3)
    optimizer_K = optim.Adam([K_param], lr=lr_K)

    criterion = nn.MSELoss()
    best_loss = float("inf")

    for epoch in range(num_epochs):
        g_model.train()
        total_loss = 0.0

        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer_g.zero_grad()

            gx, ldj_x = g_model(xb)
            gy = g_model(yb)[0]

            gx_pred = gx @ K_param
            loss = criterion(gx_pred, gy) - lamb * ldj_x / xb.numel()
            loss.backward()
            optimizer_g.step()
            spectral_norm_g_model(g_model)

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(dataset)

        g_model.eval()
        with torch.no_grad():
            gx_all = g_model(x_all)[0].detach()
            gy_all = g_model(y_all)[0].detach()

        for _ in range(100):  # K优化步数
            optimizer_K.zero_grad()
            gx_pred = gx_all @ K_param
            loss_K = criterion(gx_pred, gy_all)
            loss_K.backward()
            optimizer_K.step()

            with torch.no_grad():
                radius = spectral_radius(K_param.data)
                if radius > 1.0:
                    K_param.data /= radius

        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.6f} | K radius: {spectral_radius(K_param.data):.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_path = os.path.join(save_dir, f"{model_name}.pth")
            K_path = os.path.join(save_dir, f"{model_name}_K.pt")
            torch.save(g_model.state_dict(), model_path)
            torch.save(K_param.data, K_path)
            print(f"Saved model and K to {save_dir}")

    return g_model, K_param.data

if __name__ == '__main__':
    dim = 1 
    hidden_dim = 20
    input_dim = 0
    n_blocks = 20  
    n_layers = 1
    n_feature = 9
    rank = 5
    batch_size = 16
    n_train = 1000
    n_valid = 100
    n_test = 100
    num_epochs = 100  
    dropout = 0
    lamb = 1e-3
    learning_rate = 1e-3
    lr_g = 1e-5 
    lr_K = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = pd.read_csv('sin_train.csv', header=None).values
    X_valid = pd.read_csv('sin_valid.csv', header=None).values
    X_test = pd.read_csv('sin_test.csv', header=None).values

    length = X_train.shape[1] // n_train
    H_train = []
    for i in range(n_train):
        H_train.append(X_train[:, i*length:(i+1)*length])
    H_train = np.stack([H_train[idx].T for idx in range(n_train)], axis=0)

    X_result = np.concatenate([X_train, X_test, X_valid], axis=-1)
    Xmax = torch.tensor(np.max(X_result, axis=-1), dtype=torch.float)
    Xmin = torch.tensor(np.min(X_result, axis=-1), dtype=torch.float)
    inn_model = InvertibleNN(dim=dim+n_feature, hidden_dim=hidden_dim, n_blocks=n_blocks,
                            n_layers=n_layers, input_dim=input_dim, dropout=dropout, LDJ=lamb > 0)
    model = CombinedNetwork(inn_model=inn_model, input_dim=dim, lifted_dim=n_feature,
                            Xmax=Xmax, Xmin=Xmin)

    g_model = load_g_model("model_checkpoint_sin.ckpt", model)
    g_model, K = train_alternating(g_model, H_train, dim, dim+n_feature, num_epochs, batch_size, lr_g, lr_K, lamb, save_dir="./", model_name="sin")
