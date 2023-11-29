import torch
from torch.distributions import MultivariateNormal
from denoising_diffusion_pytorch import Unet1DCFG, GaussianDiffusion1DConditional, Trainer1DConditional, Dataset1DConditional, IterativeTrainer
import matplotlib.pyplot as plt

result_dir = './results_synthetic_iterative'

# create data
Ndata = 1000
px = MultivariateNormal(torch.zeros(2), torch.eye(2))
data = px.sample((Ndata,))
#robustness = lambda x: torch.min(torch.abs(x[:, 0]), x[:, 1])
robustness = lambda x: torch.minimum(torch.min(torch.abs(x[:, 0]), x[:, 1]), torch.tensor(3.0))
cond = torch.stack([robustness(data), robustness(data)], dim=1)

# save raw data
torch.save(data, '{}/data.pt'.format(result_dir))

print(data.shape)
print(robustness(data).shape)
print(cond.shape)

data = data.unsqueeze(-1)
cond += 4.0#torch.abs(torch.min(cond))

data = data.swapaxes(1, 2)
print(data.shape)

plt.scatter(data[:, :, 0], data[:, :, 1], c=cond[:, 0])
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.colorbar()  # to show the color scale
plt.savefig('./results_synthetic_iterative/data.png')


def callback(samples, milestone):
    print(samples)

model = Unet1DCFG(
    dim = 64,
    dim_mults = (1, 2),
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



iterative_trainer = IterativeTrainer(
    diffusion,
    rho_target = 7.0,
    alpha = 0.7,
    N = 1000,
    px = px,
    # robustness = lambda x: torch.min(torch.abs(x[:, 0]), x[:, 1]) + 4.0,
    robustness = lambda x: torch.minimum(torch.min(torch.abs(x[:, 0]), x[:, 1]), torch.tensor(3.0)) + 4.0,
    # rho_upper_bound = 7.5,
    rho_upper_bound = 7.1,
    dataset = dataset,
    train_batch_size = 32,
    train_lr = 1e-5,
    train_num_steps = 1000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    save_and_sample_every = 2000,
    num_samples=1,
    sample_conds=[2, 3, 4],
    results_folder = result_dir,
    sample_callback = callback
)
iterative_trainer.train()

