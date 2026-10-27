################################################################################################
# Contain General Model classes and functions
################################################################################################

import numpy as np
from itertools import product

from sklearn.model_selection import train_test_split

import torch
from torch import nn

from torch.utils.data import TensorDataset, DataLoader
import random
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample


def set_seed(random_state):
    """Seed Python, NumPy and PyTorch (CPU + CUDA/MPS) RNGs."""
    if random_state is None:
        return
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)          # also seeds CUDA/MPS generators
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

# Classes

class LinearPredictor(nn.Module):
    def __init__(self, in_features, intercept=True, random_state = None):
        super().__init__()
        set_seed(random_state)
        self.linear = nn.Linear(in_features, 1, bias=intercept)

    def forward(self, X):
        return self.linear(X).squeeze(-1)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_activation_nodes, hidden_activations = nn.Tanh, random_state = None):
        '''
        input_feature (int): number of features passed into the net
        hidden_activation_nodes (list of int): list containing number of nodes per hidden layer. For a K layer net, this should be have length K
        hidden_activations (list of nn class, optional): list containing activation between hidden layers (e.g nn.ReLU). For a K layer net, this should be have length K - 1
        '''
        super().__init__()

        set_seed(random_state)

        layers = []

        if type(hidden_activations) == list:
            hidden_layers_activations_tuple = list(zip(hidden_activation_nodes, hidden_activations))
        else:
            hidden_layers_activations_tuple = list(product(hidden_activation_nodes, [hidden_activations]))

        # construct hidden nodes
        for i in range(0, len(hidden_activation_nodes)):
            current_layer_node, current_activation_func = hidden_layers_activations_tuple[i]
            if i == 0:
                layers.append(
                    nn.Linear(in_features= in_features, out_features= current_layer_node, bias = True)
                )
            else:
                layers.append(
                    nn.Linear(in_features= hidden_layers_activations_tuple[i - 1][0], out_features= current_layer_node, bias = True)
                )
            layers.append(
                    current_activation_func()
                )
        # final layer
        layers.append(
            nn.Linear(hidden_activation_nodes[-1], out_features= 1)
        )
        self.net = nn.Sequential(*layers)
    def forward(self, X):
        return self.net(X).squeeze(-1)

class CANN(nn.Module):
    def __init__(self, in_features, neural_net_params, glm_bias = True, random_state = None):
        '''
        nn_params: dictionary containing paramters of the neural network (i.e hidden layers nodes)
        '''
        super().__init__()
        self.glm = LinearPredictor(in_features = in_features, intercept= glm_bias, random_state= random_state)
        self.neural_net = MLP(in_features= in_features, **neural_net_params, random_state= random_state)

        # Setting the last layer's weights to 0
        nn.init.zeros_(self.neural_net.net[-1].weight)
        nn.init.zeros_(self.neural_net.net[-1].bias)
    def forward(self,X):
        return self.glm(X) + self.neural_net(X)

class BayesianLinearPredictor(PyroModule):
    def __init__(self, in_features, out_features = 1):
        super().__init__()
        self.layer = PyroModule[nn.Linear](in_features = in_features, out_features = out_features)
        self.layer.weight = PyroSample(
            dist.Normal(0,1).expand([1, in_features]).to_event(2)
        )
        self.layer.bias = PyroSample(
            dist.Normal(0,1).expand([1]).to_event(1)
        )
    def forward(self, X):
        return self.layer(X).squeeze(-1)

class BayesianMLP(PyroModule):
    def __init__(self, in_features, hidden_activation_nodes, hidden_activations = nn.Tanh, prior_scale = 1):
        super().__init__()
        layers = []

        if type(hidden_activations) == list:
            hidden_layers_activations_tuple = list(zip(hidden_activation_nodes, hidden_activations))
        else:
            hidden_layers_activations_tuple = list(product(hidden_activation_nodes, [hidden_activations]))

        for i in range(len(hidden_activation_nodes)):
            current_layer_nodes, current_activation_func = hidden_layers_activations_tuple[i]

            # construct hidden nodes
            if i == 0:
                layer = PyroModule[nn.Linear](in_features, current_layer_nodes)
                layer.weight = PyroSample(
                    dist.Normal(0, prior_scale/in_features).expand([current_layer_nodes, in_features]).to_event(2)
                )
                layer.bias = PyroSample(
                    dist.Normal(0, prior_scale/in_features).expand([current_layer_nodes]).to_event(1)
                )
                layers.append(layer)
                layers.append(current_activation_func())

            else:
                prev_layer_nodes = hidden_layers_activations_tuple[i-1][0]

                layer = PyroModule[nn.Linear](prev_layer_nodes, current_layer_nodes)
                layer.weight = PyroSample(
                    dist.Normal(0, prior_scale/prev_layer_nodes).expand([current_layer_nodes, prev_layer_nodes]).to_event(2)
                )
                layer.bias = PyroSample(
                                    dist.Normal(0, prior_scale/in_features).expand([current_layer_nodes]).to_event(1)
                                )
                layers.append(layer)
                layers.append(current_activation_func())

        # final layer
        last_layer_nodes = hidden_activation_nodes[-1]
        layer = PyroModule[nn.Linear](last_layer_nodes, 1)
        layer.weight = PyroSample(
            dist.Normal(0, prior_scale/last_layer_nodes).expand([1, last_layer_nodes]).to_event(2)
        )
        layer.bias = PyroSample(
                            dist.Normal(0, prior_scale/last_layer_nodes).expand([1]).to_event(1)
                        )
        layers.append(layer)
        self.net = PyroModule[nn.Sequential](*layers)
    def forward(self,X):
        return self.net(X).squeeze(-1)

class BayesianCANN(PyroModule):
    def __init__(self, in_features, neural_net_params):
        super().__init__()
        self.linear_pred = BayesianLinearPredictor(in_features= in_features)
        self.neural_net = BayesianMLP(in_features = in_features, **neural_net_params)

        nn.init.zeros_(self.neural_net.net[-1].weight)
        nn.init.zeros_(self.neural_net.net[-1].bias)

    def forward(self ,X):
        return self.linear_pred(X) + self.neural_net(X)

class DeviceLoader:
    def __init__(self, *tensors, batch_size, shuffle=False, device=None):
        self.device = torch.device(device) if device is not None else torch.get_default_device()  # torch >= 2.3
        self.dataset = TensorDataset(*[t.to(self.device) for t in tensors])
        self.bs, self.shuffle = batch_size, shuffle

    def __len__(self):
        return -(-len(self.dataset) // self.bs)

    def __iter__(self):
        n = len(self.dataset)
        idx = (torch.randperm(n, device=self.device) if self.shuffle
               else torch.arange(n, device=self.device))
        T = self.dataset.tensors
        for i in range(0, n, self.bs):
            j = idx[i:i + self.bs]
            yield tuple(t[j] for t in T)
            
# Functions

def poisson_deviance(y_true, y_pred, eps=1e-8):
    """
    y_pred: eta, not actualy y (used y_pred for consistency with other torch.loss instances)
    """
    mu = torch.exp(y_pred.clamp(max=20))      # clamp max to avoid overflow
    log_term = torch.where(
        y_true > 0,
        y_true * (torch.log(y_true.clamp(min=eps)) - y_pred),   # log(y/mu) = log y - eta
        torch.zeros_like(y_true),
    )
    return 2 * (log_term - (y_true - mu)).mean()

def mlp_train_loop(model, optimizer, loss_fn, n_epochs, train_loader, val_loader,
                   device="cpu", patience=20, min_delta=1e-9, verbose_every=100):
    model.to(device)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state, best_epoch = None, 0
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        ### Training
        model.train()
        train_loss, n_train = 0.0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model(X_batch)
            loss = loss_fn(y_pred=y_pred, y_true=y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            n_train += len(y_batch)
        train_loss /= n_train

        ### Validation
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.inference_mode():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                val_pred = model(X_batch)
                batch_loss = loss_fn(y_pred=val_pred, y_true=y_batch)

                val_loss += batch_loss.item() * len(y_batch)
                n_val += len(y_batch)
        val_loss /= n_val

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        ### Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % verbose_every == 0 or epoch == 0:
            print(f"Epoch {epoch + 1} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}. "
                  f"Best val loss {best_val_loss:.4f} at epoch {best_epoch}")
            break

    ### Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }

def closure_cann(model, optimiser, loss_func, X, y):
    def closure():
        optimiser.zero_grad()
        eta = model(X)
        loss = loss_func(eta, y)
        loss.backward()
        return loss
    return closure


def cann_training_loop(cann_model, glm_loss_function, nn_loss_function,
                       nn_optimiser, nn_optim_params, nn_epochs,
                       train_loader, val_loader, early_stop_params =None, device="cpu", random_state = None):
    set_seed(random_state)
    cann_model.to(device)
    X_all, y_all = train_loader.dataset.tensors
    X_all, y_all = X_all.to(device), y_all.to(device)

    # Step 1: fit GLM alone
    glm_optimiser = torch.optim.LBFGS(
        cann_model.glm.parameters(), lr=1.0, max_iter=2000,
        line_search_fn="strong_wolfe")
    
    glm_optimiser.step(closure_cann(cann_model.glm, glm_optimiser,
                                    glm_loss_function, X_all, y_all))

    # Step 2: freeze GLM
    for p in cann_model.glm.parameters():
        p.requires_grad = False

    # Step 3: train network on residual structure (GLM + NN output)
    optimiser = nn_optimiser(cann_model.neural_net.parameters(), **nn_optim_params)
    if early_stop_params is not None:

        history = mlp_train_loop(cann_model, optimiser, nn_loss_function,
                             nn_epochs, train_loader, val_loader, device=device, **early_stop_params)
    else:
        history = mlp_train_loop(cann_model, optimiser, nn_loss_function,
                                     nn_epochs, train_loader, val_loader, device=device)
    return history