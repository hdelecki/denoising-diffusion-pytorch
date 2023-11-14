import torch
from denoising_diffusion_pytorch import Unet1DConditional, GaussianDiffusion1DConditional, Trainer1DConditional, Dataset1DConditional

model = Unet1DConditional(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 32
)

diffusion = GaussianDiffusion1DConditional(
    model,
    seq_length = 32,
    timesteps = 1000,
    objective = 'pred_v'
)

noise_scale = 0.2
train1 = torch.rand(64, 32, 32)*noise_scale # features are normalized from 0 to 1
train1[:, :16, :16] = 0.9

train2 = torch.rand(64, 32, 32)*noise_scale # features are normalized from 0 to 1
train2[:, 16:, 16:] = 0.9

train3 = torch.rand(64, 32, 32)*noise_scale # features are normalized from 0 to 1
train3[:, :16, 16:] = 0.9

train4 = torch.rand(64, 32, 32)*noise_scale # features are normalized from 0 to 1
train4[:, 16:, :16] = 0.9

training_seq = torch.cat((train1, train2, train3, train4), 0)

#training_seq = torch.cat((train1, train2), 0)



cscales = [-1.0, -0.33, 0.33, 1.0]
conditions1 = cscales[0]*torch.ones(64, 32, 32)
conditions2 = cscales[1]*torch.ones(64, 32, 32)
conditions3 = cscales[2]*torch.ones(64, 32, 32)
conditions4 = cscales[3]*torch.ones(64, 32, 32)

conditions = torch.cat((conditions1, conditions2, conditions3, conditions4), 0)
#conditions = torch.cat((conditions1, conditions4), 0)


print(training_seq.shape)
print(conditions.shape)

dataset = Dataset1DConditional(training_seq, conditions)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

#loss = diffusion(training_seq, conditions)
# loss.backward()

# Or using trainer

trainer = Trainer1DConditional(
    diffusion,
    dataset = dataset,
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 50000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    save_and_sample_every = 100,
    sample_conds=cscales,
    results_folder = './results_conditional',
)
trainer.train()
# trainer.load(7)

# # after a lot of training

# sampled_seq = diffusion.sample(batch_size = 4)