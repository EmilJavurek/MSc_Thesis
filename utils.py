import torch
from torch.utils.data import Dataset 
from torch.nn import Sequential, Linear, SELU, Dropout
import matplotlib.pyplot as plt
import numpy as np


# Flow matching objective:

def time_uniform(n):
    return torch.rand(n)

def power_law_(n,alpha):
    """uniform as alpha --> +inf (or -inf)
    for alpha > -1 it gives preference to higher t
    for alpha < -2 it gives preference to smaller t
    
    Use to create t-sampling distribution as follows:

    power_law = partial(power_law_, alpha = -0.5)
    """
    return torch.rand(n)**((1+alpha)/(2+alpha))


def FM_(x1, sigma_min, batch, device, t_dist = time_uniform):
    """
    Calculates the Flow Matching objective for a batch of points from target (that we condition on)
    
    Use to create:
    FM = partial(FM_, --here specify all inputs except x1--)
    And in actual training loop use FM()
    """
    t = t_dist(batch).view([batch] +  [1] * (x1.dim() -1))
    mu_t = t * x1
    sigma_t = 1 - (1-sigma_min)*t
    eps = torch.randn_like(x1)
    xt = mu_t + sigma_t * eps
    ut = (x1 - (1 - sigma_min) * xt) / (1 - (1 - sigma_min) * t)
    input = torch.cat([xt, t], dim = -1).to(device)
    return input, ut


# NN parameterization of v:

class MLP(torch.nn.Module):
    """For toy generative problems"""                                         
    def __init__(self, dim=2, w=64):
        super().__init__()
        self.net = Sequential(
            Linear(dim + 1, w), # +1 for time
            SELU(),
            Linear(w, w),
            SELU(),
            Linear(w, w),
            SELU(),
            Linear(w, dim),
        )

    def forward(self, x):
        return self.net(x)
    
class MLP_FMPE(torch.nn.Module):
    """For FMPE
    input [x,theta_t,t] 
    output [v] (same dim as theta_t)
    """                                         
    def __init__(self, input_dim, output_dim, w=64):
        super().__init__()
        self.net = Sequential(
            Linear(input_dim, w),
            SELU(),
            Linear(w, w),
            SELU(),
            Linear(w, w),
            SELU(),
            Linear(w, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# Dataset object

class SbiDataset(Dataset):
    """For inference problems - stores the (theta,x) joint sample"""
    def __init__(self, prior, simulator, n, scaling = 2):
        self.simulator = simulator
        self.prior = prior
        self.n = n
        self.scaling = scaling #Wildeberger & Dax have =1

        self.theta = self.prior(self.n)
        self.x = self.simulator(self.theta)
        
        #standardize
        self.mean = {"x": self.x.mean(dim = 0).numpy(),
                     "theta": self.theta.mean(dim = 0).numpy()}
        self.std = {"x": self.x.std(dim=0).numpy(),
                    "theta": self.theta.std(dim=0).numpy()}
        
        self.theta = self.standardize(self.theta, label = "theta")
        self.x = self.standardize(self.x, label= "x")

    def standardize(self, data, label, inverse = False):
        if not inverse:
            return self.scaling*(data - self.mean[label])/self.std[label]
        else:
            return data * self.std[label]/self.scaling + self.mean[label]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.cat([self.theta[idx], self.x[idx]], dim = -1)    



class ToysDataset(Dataset):
    """For toy generative problems"""
    def __init__(self, target, dim, n, **dist_args):
        self.dim = dim 
        self.n = n
        self.data = target(n = self.n, **dist_args)
        self.mean = self.data.mean(axis=0).numpy()
        self.std = self.data.std(axis=0).numpy()
        self.data = self.standardize(self.data)

    def standardize(self, data, inverse = False):
        #scaling by 2 experimentally favorable
        if not inverse:
            return 2*(data - self.mean)/self.std
        else:
            return data * self.std/2 + self.mean

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.data[idx]



# Plotting trajectories

class torch_wrapper_FMPE(torch.nn.Module):
    """
    Make .forward compatible with NeuralODE object.
    Adapted from torchcfm library.
    FOR FMPE INFERENCE MODELS
    assuming obs is 1 observation!
    
    input: [theta]
    output: model([x_obs,theta,t]) which outputs in the space of theta
    """
    def __init__(self, model, obs):
        super().__init__()
        self.model = model
        self.obs = obs

    def forward(self, t, theta, *args, **kwargs):
        rep = theta.shape[0]
        return self.model(torch.cat([self.obs.repeat(rep,1),theta,t.repeat(rep,1)], dim = -1)) #[x,theta_t,t]


class torch_wrapper(torch.nn.Module):
    """
    Make .forward compatible with NeuralODE object.
    Adapted from torchcfm library.
    FOR GENERATIVE MODELS
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        """ 
        Inputs:
        t       :       torch.tensor(float)
        x       :       (batch,*dim) tensor
        """
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))

def plot_trajectories(traj, box = None, target = None, n = None):                                        
    """
    Adapted from torchcfm library (own code)
    Plot trajectories of some selected samples.
    traj is (t,batch,*dim) tensor

    Make sure to select only 2 dimensions from *dim
    """
    n = traj.shape[1] if not n else n
    plt.figure(figsize=(12, 12))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.4, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
    plt.legend(["Prior", "Flow", "Result"])

    #Target density outline
    if target is not None:
        plt.scatter(target[:,0],target[:,1],s = 1, alpha = 0.1, color = "red")
        plt.legend(["Prior", "Flow", "Result", "Target"])


    #Fixing plot axis lengths:
    if box and isinstance(box, (list,tuple)):
        plt.xlim(box[0])
        plt.ylim(box[1])
    elif box and isinstance(box, int):
        plt.xlim(-box,box)
        plt.ylim(-box,box)
    else:
        s = 1.2
        plt.xlim(s*traj[:,:,0].min(),s*traj[:,:,0].max())
        plt.ylim(s*traj[:,:,1].min(),s*traj[:,:,1].max())

    plt.grid(True)
    plt.show()



# Diagnostic plots

def plot_trainloss(losses: list, **kwargs):
    #training steps logloss
    print(f"final training loss: {losses[-1]:.3f}, last 100 avg: {np.mean(losses[-100:]):.3f}")
    plt.plot(range(len(losses)),np.log10(losses))
    plt.title("Training Log loss")
    plt.xlabel("training step")
    return plt

def plot_epochloss(epoch_tr_loss: list, epoch_val_loss: list, **kwargs):
    #epoch losses
    print(f"final train loss: {epoch_tr_loss[-1]:.3f} (10 avg: {np.mean(epoch_tr_loss[-10:]):.3f}), \n final val loss: {epoch_val_loss[-1]:.3f} (10 avg: {np.mean(epoch_val_loss[-10:]):.3f})")
    plt.plot(range(len(epoch_tr_loss)), np.log10(epoch_tr_loss), label = "training loss")
    plt.plot(range(len(epoch_tr_loss)), np.log10(epoch_val_loss), label = "validation loss")
    plt.title("Epoch log losses")
    plt.xlabel("epoch")
    plt.legend()
    return plt