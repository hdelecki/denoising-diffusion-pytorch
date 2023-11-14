import json
import torch
from denoising_diffusion_pytorch import Unet1DConditional, GaussianDiffusion1DConditional, Trainer1DConditional, Dataset1DConditional



# load data
with open('dummy_data.json', 'r') as f:
    data = json.load(f)

disturbances = torch.tensor(data['disturbances'])

print(disturbances.shape)
Ndata, horizon, xdim = disturbances.shape

rewards = torch.tensor(data['rewards'])
print(rewards.shape)
print(rewards.min())
print(rewards.mean())
print(rewards.std())
print(rewards.max())
conds = rewards.view(-1, 1, 1) * torch.ones((Ndata, horizon, xdim))

disturbances = disturbances.swapaxes(1, 2)
conds = conds.swapaxes(1, 2)

print(disturbances.shape)
print(conds.shape)



model = Unet1DConditional(
    dim = 64,
    dim_mults = (1, 2, 4),
    channels = 2
)

diffusion = GaussianDiffusion1DConditional(
    model,
    seq_length = 100,
    timesteps = 1000,
    objective = 'pred_v'
)



# print(training_seq.shape)
# print(conditions.shape)

dataset = Dataset1DConditional(disturbances, conds)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

#loss = diffusion(training_seq, conditions)
# # loss.backward()

# # Or using trainer

trainer = Trainer1DConditional(
    diffusion,
    dataset = dataset,
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 50000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    save_and_sample_every = 1000,
    sample_conds=[5.0],
    results_folder = './results_rw',
)
trainer.load(14)


cond_eval  = 1.0 * torch.ones(1, 2, 100).to('cuda')
sampled_seq = diffusion.sample(cond_eval, batch_size = 1)
sampled_seq.shape
print(sampled_seq.shape)
print(sampled_seq[0, :, :].sum(dim=1))

# import matplotlib.pyplot as plt
# plt.imshow(sampled_seq[0].permute(1, 0).cpu().numpy())
# plt.savefig('sampled_seq.png')
