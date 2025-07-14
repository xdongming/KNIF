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

class ResidualFlow(nn.Module):
    def __init__(self, dim, hidden_dim, input_dim=0, dropout=0, LDJ=False):
        super(ResidualFlow, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.LDJ = LDJ
        self.dropout = dropout
        self.n_layers = 1
        
        layers = [nn.Linear(self.dim + self.dim * (self.input_dim > 0), self.hidden_dim), nn.ReLU()]
        for _ in range(self.n_layers):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(self.hidden_dim, self.dim))
        self.net = nn.Sequential(*layers)
        if self.input_dim > 0:
            self.cheby = nn.Linear(self.input_dim, self.dim - self.input_dim)
        self._initialize_weights()
    
    def forward(self, x, u=None, reverse=False):
        def func(x_):
            x_e = torch.cat((x_, u_), dim=-1) if u is not None else x_
            return self.net(x_e)
        if u is not None:
            u = F.tanh(u) / (1+1e-3)
            chebyshev = torch.cos(self.cheby(torch.arccos(u)))
            u = torch.cat((u, chebyshev), dim=-1)
        if not reverse:   
            y = x + func(x)
            if self.LDJ:
                x = x.view(-1, x.shape[-1])
                u = u.view(-1, u.shape[-1]) if u is not None else None
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
            epsilon = 1e-6
            det = 1
            max_iter = 1000
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
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'cheby' in name:
                    lambda_s = 5
                    module.weight.data = torch.distributions.exponential.Exponential(lambda_s).sample(module.weight.shape)
                else:
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

class InvertibleNN(nn.Module):
    def __init__(self, dim, hidden_dim, n_blocks, input_dim=0, dropout=0, LDJ=False):
        super(InvertibleNN, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.input_dim = input_dim
        self.blocks = nn.ModuleList([ResidualFlow(self.dim, self.hidden_dim, self.input_dim, dropout, LDJ) for _ in range(self.n_blocks)])
    
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
        self.K = nn.Parameter(torch.randn(input_dim + lifted_dim, input_dim + lifted_dim), requires_grad=True)
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
        
def dmd(model, X):
    GX_pred_list = []
    GX_list = []
    GX, ldj = model(X)
    for i in range(X.shape[0]):
        GX_temp = GX[i, :, :].T
        GX_pred = model.K @ GX_temp[:, :-1]
        GX_pred_list.append(GX_pred)
        GX_list.append(GX_temp[:, 1:])
    GX_pred = torch.cat(GX_pred_list, dim=-1)
    GX = torch.cat(GX_list, dim=1)

    return GX, GX_pred, ldj

class TrainModel(pl.LightningModule):
    def __init__(self, model, train_dataset, learning_rate=1e-3, lamb=1, path="model_checkpoint_NP"):
        super(TrainModel, self).__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        self.best_val_loss = float('inf')  
        self.validation_outputs = []
        self.lamb = lamb
        self.train_losses = []
        self.path = path+'.ckpt'

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X_batch = batch[0]
        GY, GY_pred, ldj = dmd(self.model, X_batch)

        loss_lin = self.criterion(GY, GY_pred.detach())
        loss_LDJ = ldj / X_batch.numel()

        loss = loss_lin - self.lamb * loss_LDJ
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        Z_batch = batch[0]
        Z1, Z_pred, _ = dmd(self.model, Z_batch)
        Z_pred = self.model(Z_pred.T, reverse=True)
        Z1 = self.model(Z1.T, reverse=True)
        valid_loss = self.criterion(Z_pred, Z1)

        self.validation_outputs.append(valid_loss)
        self.log('val_loss', valid_loss)
        return valid_loss

    def test_step(self, batch, batch_idx):
        Z_batch = batch[0]
        Z1, Z_pred, _ = dmd(self.model, Z_batch)
        Z_pred = self.model(Z_pred.T, reverse=True)
        Z1 = self.model(Z1.T, reverse=True)
        test_loss = self.criterion(Z_pred, Z1)

        self.log('test_loss', test_loss)
        return test_loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.no_grad():  
            for name, module in self.model.named_modules():  
                if isinstance(module, nn.Linear): 
                    if name == "linear":  
                        continue
                    weight = module.weight  
                    sigma_max = torch.linalg.norm(weight, ord=2)  
                    if sigma_max >= 1 - 1e-2:  
                        scale = (1 - 1e-2) / sigma_max
                        module.weight.data *= scale
    
    # def on_train_epoch_start(self):
    #     if os.path.exists(self.path):
    #         best_state_dict = torch.load(self.path)["state_dict"]
    #         self.load_state_dict(best_state_dict)
    
    def on_train_epoch_end(self):
        device = self.model.K.device
        self.model.eval() 
        with torch.no_grad():
            x_all = self.train_dataset.tensors[0].to(device)  

            gx_all = self.model(x_all)[0].detach()[:, :-1, :] 
            gy_all = self.model(x_all)[0].detach()[:, 1:, :]

            gx_all = gx_all.reshape(-1, gx_all.shape[-1])
            gy_all = gy_all.reshape(-1, gy_all.shape[-1])

        optimizer_K = torch.optim.Adam([self.model.K], lr=1e-3)
        for _ in range(100):  
            optimizer_K.zero_grad()
            gx_pred = gx_all @ self.model.K
            loss_K = self.criterion(gx_pred, gy_all)
            loss_K.backward()
            optimizer_K.step()
            with torch.no_grad():
                radius = torch.linalg.norm(self.model.K.data, ord=2)
                if radius > 1.0:
                    self.model.K.data /= radius

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.validation_outputs).mean() 
        self.log('avg_val_loss', avg_val_loss)
        self.validation_outputs.clear()
        print(f"Validation loss: {avg_val_loss}")
        with open("loss_log.txt", "a") as f:
            f.write(f"{avg_val_loss.item()}\n")

    def configure_optimizers(self):
        g_params = [p for n, p in self.model.named_parameters() if "K" not in n]
        optimizer = torch.optim.AdamW(g_params, lr=self.learning_rate, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100,
            gamma=0.5
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
    
if __name__ == '__main__':
    dim = 1 
    hidden_dim = 80 
    input_dim = 0
    n_blocks = 3  
    n_feature = 19
    batch_size = 64
    # n_train = 6400
    n_train = 2620
    # n_valid = 1000
    n_test = 1
    dropout = 0
    num_epochs = 1000 
    lamb = 1e-3
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = pd.read_csv('y_train.csv', header=None).values.astype(float)
    length = X_train.shape[1] // n_train
    H_train = []
    for i in range(n_train):
        H_train.append(X_train[:, i*length:(i+1)*length])
    H_train = np.stack([H_train[idx].T for idx in range(n_train)], axis=0)
    train_dataset = TensorDataset(torch.tensor(H_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    X_result = np.concatenate([X_train], axis=-1)
    Xmax = torch.tensor(np.max(X_result, axis=-1), dtype=torch.float)
    Xmin = torch.tensor(np.min(X_result, axis=-1), dtype=torch.float)

    warnings.filterwarnings("ignore")
    path = "model_checkpoint_Wind"
    checkpoint_callback = ModelCheckpoint(
        # monitor="avg_val_loss", 
        monitor="train_loss", 
        dirpath="./",  
        filename=path, 
        save_top_k=1, 
        mode="min",    
    )
    inn_model = InvertibleNN(dim=dim+n_feature, hidden_dim=hidden_dim, n_blocks=n_blocks, input_dim=input_dim, dropout=dropout, LDJ=lamb>0)
    model = CombinedNetwork(inn_model=inn_model, input_dim=dim, lifted_dim=n_feature, Xmax=Xmax, Xmin=Xmin)
    lightning_model = TrainModel(model=model, train_dataset=train_dataset, learning_rate=learning_rate, lamb=lamb, path=path)
    trainer = pl.Trainer(accelerator="gpu", devices=4, strategy="ddp_find_unused_parameters_true", max_epochs=num_epochs, callbacks=[checkpoint_callback])

    trainer.fit(lightning_model, train_loader, train_loader)