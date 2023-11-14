import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D

model = Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 32
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 32,
    timesteps = 1000,
    objective = 'pred_v'
)

training_seq = torch.rand(64, 32, 32) # features are normalized from 0 to 1
training_seq[:, 16, :] = 0.9
dataset = Dataset1D(training_seq)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

# loss = diffusion(training_seq)
# loss.backward()

# Or using trainer

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 32,
    train_lr = 1e-4,
    train_num_steps = 7000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
)
# trainer.train()
trainer.load(7)

# after a lot of training

sampled_seq = diffusion.sample(batch_size = 4)
sampled_seq.shape # (4, 32, 128)
print(sampled_seq.shape)

import matplotlib.pyplot as plt
plt.imshow(sampled_seq[0].permute(1, 0).cpu().numpy())
plt.savefig('sampled_seq.png')
