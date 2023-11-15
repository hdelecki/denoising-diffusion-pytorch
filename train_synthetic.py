import torch
from torch.distributions import MultivariateNormal
from denoising_diffusion_pytorch import Unet1DCFG, GaussianDiffusion1DConditional, Trainer1DConditional, Dataset1DConditional
import matplotlib.pyplot as plt

# create data
Ndata = 512
px = MultivariateNormal(torch.zeros(2), torch.eye(2))
data = px.sample((Ndata,))
robustness = lambda x: torch.min(torch.abs(x[:, 0]), x[:, 1])
cond = torch.stack([robustness(data), robustness(data)], dim=1)
print(data.shape)
print(robustness(data).shape)
print(cond.shape)

data = data.unsqueeze(-1)
cond += torch.abs(torch.min(cond))

data = data.swapaxes(1, 2)
print(data.shape)

plt.scatter(data[:, :, 0], data[:, :, 1], c=cond[:, 0])
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.colorbar()  # to show the color scale
plt.savefig('./results_synthetic/data.png')


def callback(samples, milestone):
    print(samples)

model = Unet1DCFG(
    dim = 64,
    dim_mults = (1,),
    channels = 1,
    cond_dim = 2,
    cond_drop_prob = 0.1,
)

diffusion = GaussianDiffusion1DConditional(
    model,
    seq_length = 2,
    timesteps = 1000,
    objective = 'pred_v',
    auto_normalize=False,
    clip_min=-5.0,
    clip_max=5.0,
)

dataset = Dataset1DConditional(data, cond)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below



trainer = Trainer1DConditional(
    diffusion,
    dataset = dataset,
    train_batch_size = 32,
    train_lr = 1e-4,
    train_num_steps = 50000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    save_and_sample_every = 1000,
    num_samples=1,
    sample_conds=[1.0, 2.0, 3.0, 4.0],
    results_folder = './results_synthetic',
    sample_callback = callback
)
trainer.train()