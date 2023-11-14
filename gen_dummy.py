import json
import torch
from torch.distributions import Normal, MultivariateNormal



xdim = 2
horizon = 100

px = Normal(torch.zeros(xdim), torch.ones(xdim))

l2_norm = lambda x: torch.norm(x.sum(dim=0), p=2)


Ndata = 100
# data_dicts = []

xdata = torch.zeros(Ndata, horizon, xdim)
rdata = torch.zeros(Ndata)

for i in range(Ndata):
    x = px.sample((horizon,))
    xdata[i] = x
    r = l2_norm(x)
    rdata[i] = r

print(rdata.min())
print(rdata.mean())
print(rdata.std())
print(rdata.max())



combined = {'disturbances': xdata.tolist(), 'rewards': rdata.tolist()}
with open('dummy_data.json', 'w') as f:
    json.dump(combined, f)