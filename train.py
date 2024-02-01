import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
from datetime import datetime

current_date = datetime.now()

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print("")
print(Colors.BOLD + "Polaris LLM Training v0.1.0")
print("------------------------------------------")
print(f"Training started: {current_date}" + Colors.ENDC)
print("")

lines = open('input.txt', 'r').read()

vocab = sorted(list(set(lines)))

tags = ['<START>', '<END>', '<TXAI>', '<USER>']
for tag in tags:
    vocab.append(tag)

itos = {i:ch for i, ch in enumerate(vocab)}
stoi = {ch:i for i, ch in enumerate(vocab)}

# simple tokenization by characters
def encode(s):
    return [stoi[ch] for ch in s]

def decode(l):
    return ''.join([itos[i] for i in l])

MASTER_CONFIG = {
    "vocab_size": len(vocab),
}

dataset = torch.tensor(encode(lines), dtype=torch.int32)
dataset.shape

#########

def get_batches(data, split, batch_size, context_window, config=MASTER_CONFIG):
    train = data[:int(.8 * len(data))]
    val = data[int(.8 * len(data)): int(.9 * len(data))]
    test = data[int(.9 * len(data)):]
    
    batch_data = train
    if split == 'val':
        batch_data = val

    if split == 'test':
        batch_data = test
    
    # pick random Running points
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))
    x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()
    return x, y

MASTER_CONFIG.update({
    'batch_size': 24,
    'context_window': 32,
    'opt_adam_lr': 0.0002
})

xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

[(decode(xs[i].tolist()), decode(ys[i].tolist())) for i in range(len(xs))]

@torch.no_grad()  # don't compute gradients for this function
def evaluate_loss(model, config=MASTER_CONFIG):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = []
        for _ in range(10):
            xb, yb = get_batches(dataset, split, config['batch_size'], config['context_window'])
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out

class SimpleBrokenModel(nn.Module):
    def __init__(self, config=MASTER_CONFIG):
        super().__init__()
        self.config = config
        print(Colors.WARNING + "Running SimpleBrokenModel" + Colors.ENDC)

        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
            nn.Linear(config['d_model'], config['vocab_size']),
        )
        print(Colors.RED + Colors.BOLD)
        print("model params from SimpleBrokenModel:", sum([m.numel() for m in self.parameters()]))
        print(Colors.ENDC)

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        a = self.linear(x)
        logits = F.softmax(a, dim=-1)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

        else:
            return logits

MASTER_CONFIG.update({
    'd_model': 128,
})

print(Colors.OKGREEN + "### MASTER_CONFIG SimpleBrokenModel 01 ###" + Colors.ENDC)
print(Colors.OKGREEN + str(MASTER_CONFIG) + Colors.ENDC)  
print(Colors.OKGREEN + "###" + Colors.ENDC)
print("")

model = SimpleBrokenModel(MASTER_CONFIG)
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

logits, loss = model(xs, ys)

MASTER_CONFIG.update({
    'epochs': 1000,
    'log_interval': 10,
    'batch_size': 24,
})

print(Colors.OKGREEN + "### MASTER_CONFIG SimpleBrokenModel 02 ###" + Colors.ENDC)
print(Colors.OKGREEN + str(MASTER_CONFIG) + Colors.ENDC)  
print(Colors.OKGREEN + "###" + Colors.ENDC)
print("")

model = SimpleBrokenModel(MASTER_CONFIG)

optimizer = torch.optim.Adam(model.parameters(),MASTER_CONFIG['opt_adam_lr'])

def train(model, optimizer, scheduler=None, config=MASTER_CONFIG, print_logs=True):
    losses = []
    start_time = time.time()
    print(Colors.BOLD + "Training function started at:", datetime.now())
    print(Colors.ENDC)
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        
        xs, ys = get_batches(dataset, 'train', config['batch_size'], config['context_window'])
        
        # Start timer for forward and backward pass
        forward_start = time.time()
        logits, loss = model(xs, targets=ys)
        loss.backward()
        optimizer.step()
        forward_end = time.time()
        
        if scheduler:
            scheduler.step()
        
        if epoch % config['log_interval'] == 0:
            batch_time = time.time() - start_time
            x = evaluate_loss(model)
            losses += [x]
            if print_logs:
                print(Colors.OKBLUE + f"Epoch {epoch} | val loss {x['val']:.3f} | "
                    f"Time {batch_time:.3f} | "
                    f"Forward Time {forward_end - forward_start:.3f} | "
                    f"ETA in seconds {batch_time * (config['epochs'] - epoch) / config['log_interval']:.3f}" + Colors.ENDC)
            start_time = time.time()

            if scheduler:
                print("lr: ", scheduler.get_lr())

    print(Colors.BOLD)
    print("Training function ended at:", datetime.now())
    print("validation loss: ", losses[-1]['val'])
    print(Colors.ENDC)
    return pd.DataFrame(losses).plot()

train(model, optimizer)

class SimpleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        print(Colors.WARNING + "Running SimpleModel" + Colors.ENDC)

        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

        print(Colors.RED + Colors.BOLD)
        print("model params from SimpleModel:", sum([m.numel() for m in self.parameters()]))
        print(Colors.ENDC)

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        logits = self.linear(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

        else:
            return logits

model = SimpleModel(MASTER_CONFIG)
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

logits, loss = model(xs, ys)
optimizer = torch.optim.Adam(model.parameters(),MASTER_CONFIG['opt_adam_lr'])
train(model, optimizer)

def generate(model, config=MASTER_CONFIG, max_new_tokens=30):
    print("generate started at:", datetime.now())
    idx = torch.zeros(5, 1).long()
    for _ in range(max_new_tokens):
        # call the model
        logits = model(idx[:, -config['context_window']:])
        last_time_step_logits = logits[
            :, -1, :
        ]  # all the batches (1), last time step, all the logits
        p = F.softmax(last_time_step_logits, dim=-1)  # softmax to get probabilities
        idx_next = torch.multinomial(
            p, num_samples=1
        )  # sample from the distribution to get the next token
        idx = torch.cat([idx, idx_next], dim=-1)  # append to the sequence
    print("generate ended at:", datetime.now())
    return [decode(x) for x in idx.tolist()]

generate(model)

class RMSNorm(nn.Module):
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))

        print(Colors.WARNING + "Running RMSNorm" + Colors.ENDC)

    def forward(self, x):
        """
        assumes shape is (batch, seq_len, d_model)
        """
        # frob norm is not the same as RMS. RMS = 1/sqrt(N) * frob norm
        ff_rms = torch.linalg.norm(x, dim=(1,2)) * x[0].numel() ** -.5
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)
        return self.scale[:x.shape[1], :].unsqueeze(0) * raw

config = {
    'batch_size': 5,
    'context_window': 11,
    'd_model': 13,
}

print(Colors.OKGREEN + "### 'config' RMSNorm ###" + Colors.ENDC)
print(Colors.OKGREEN + str(config) + Colors.ENDC)  
print(Colors.OKGREEN + "###" + Colors.ENDC)
print("")

batch = torch.randn((config['batch_size'], config['context_window'], config['d_model']))
m = RMSNorm((config['context_window'], config['d_model']))
g = m(batch)
print(g.shape)

rms = torch.linalg.norm(batch, dim=(1,2)) * (batch[0].numel() ** -.5)

# scaled_batch.var(dim=(1,2))
assert torch.linalg.norm( torch.arange(5).float() ) == (torch.arange(5).float() ** 2 ).sum() ** .5
rms = torch.linalg.norm( torch.arange(5).float() ) * (torch.arange(5).numel() ** -.5)
assert torch.allclose(torch.linalg.norm(torch.arange(5).float() / rms), torch.tensor(5 ** .5))
ff_rms = torch.linalg.norm(batch, dim=(1,2)) * batch.shape[1:].numel() ** -.5

# RMS for sure
ffx = torch.zeros_like(batch)
for i in range(batch.shape[0]):
    ffx[i] = batch[i] / ff_rms[i]
assert torch.allclose(torch.linalg.norm(ffx, dim=(1,2)) ** 2, torch.tensor(143).float())
assert torch.allclose(ffx, g)

class SimpleModel_RMS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        print(Colors.WARNING + "Running SimpleModel_RMS" + Colors.ENDC)

        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.rms = RMSNorm((config['context_window'], config['d_model']))
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

        print(Colors.RED + Colors.BOLD)
        print("model params from SimpleModel_RMS:", sum([m.numel() for m in self.parameters()]))
        print(Colors.ENDC)

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        x = self.rms(x) # rms pre-normalization
        logits = self.linear(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

        else:
            return logits

model = SimpleModel_RMS(MASTER_CONFIG)
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

logits, loss = model(xs, ys)
optimizer = torch.optim.Adam(model.parameters(),MASTER_CONFIG['opt_adam_lr'])
train(model, optimizer)

def get_rotary_matrix(context_window, embedding_dim):
    R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
    for position in range(context_window):
        for i in range(embedding_dim//2):
            theta = 10000. ** (-2.*(i - 1) / embedding_dim)
            m_theta = position * theta
            R[position, 2*i,2*i] = np.cos(m_theta)
            R[position, 2*i,2*i+1] = - np.sin(m_theta)
            R[position, 2*i+1,2*i] = np.sin(m_theta)
            R[position, 2*i+1,2*i+1] = np.cos(m_theta)
    return R

K = 3
config = {
    'batch_size': 10,
    'd_model': 32,
    'n_heads': 8,
    'context_window': K**2,
}

print(Colors.OKGREEN + "### 'config' Rotary Matrix 01 ###" + Colors.ENDC)
print(Colors.OKGREEN + str(config) + Colors.ENDC)  
print(Colors.OKGREEN + "###" + Colors.ENDC)
print("")

batch = torch.randn(1, config['context_window'], config['d_model'])
R = get_rotary_matrix(config['context_window'], config['d_model'])
fig, ax = plt.subplots(K, K, figsize=(K * 3, K * 4))

for i in range(K):
    for j in range(K):
        ax[i, j].imshow(R[i * K + j, :, :].detach().numpy())
        ax[i, j].set_title(f'rotation at {i * K + j}')

config = {
    'd_model': 128,
    'context_window': 32,
}

print(Colors.OKGREEN + "### 'config' Rotary Matrix 02 ###" + Colors.ENDC)
print(Colors.OKGREEN + str(config) + Colors.ENDC)  
print(Colors.OKGREEN + "###" + Colors.ENDC)
print("")

R = get_rotary_matrix(config['context_window'], config['d_model'])
x = torch.randn(config['d_model'])
y = torch.randn(config['d_model'])

m = 3
n = 13

x_m = R[m,:,:] @ x
x_n = R[n,:,:] @ y

assert torch.isclose(x_m @ x_n, x @ R[n-m,:,:] @ y)

config = {
    'batch_size': 24,
    'd_model': 128,
    'n_heads': 8,
    'context_window': 32,
}

print(Colors.OKGREEN + "### 'config' RoPEAttentionHead 01 ###" + Colors.ENDC)
print(Colors.OKGREEN + str(config) + Colors.ENDC)  
print(Colors.OKGREEN + "###" + Colors.ENDC)
print("")

class RoPEAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        print(Colors.WARNING + "Running RoPEAttentionHead" )

        self.w_q = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.w_k = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.w_v = nn.Linear(config['d_model'], config['d_model'], bias=False)

        self.R = get_rotary_matrix(config['context_window'], config['d_model'])

    def get_rotary_matrix(context_window, embedding_dim):
        R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
        for position in range(context_window):
            for i in range(embedding_dim//2):
                theta = 10000. ** (-2.*(i - 1) / embedding_dim)
                m_theta = position * theta
                R[position, 2*i,2*i] = np.cos(m_theta)
                R[position, 2*i,2*i+1] = - np.sin(m_theta)
                R[position, 2*i+1,2*i] = np.sin(m_theta)
                R[position, 2*i+1,2*i+1] = np.cos(m_theta)
        return R
    
    def forward(self, x, return_attn_weights=False):
        b,m,d = x.shape
        
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q_rotated = (torch.bmm(q.transpose(0,1), self.R[:m])).transpose(0,1)
        k_rotated = (torch.bmm(k.transpose(0,1), self.R[:m])).transpose(0,1)

        activations = F.scaled_dot_product_attention(
            q_rotated,k_rotated,v,dropout_p =.1
        )

        if return_attn_weights:
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1,2)) / np.sqrt(d)
            attn_weights = F.softmax(attn_weights, dim=-1)
            return activations, attn_weights
        return activations

layer = RoPEAttentionHead(config)
batch = torch.randn((config['batch_size'], config['context_window'], config['d_model']))
output, attn_weights = layer(batch, return_attn_weights=True)

x = torch.randn((config['batch_size'], config['context_window'], config['d_model']))

q = layer.w_q(x)
k = layer.w_k(x)
v = layer.w_v(x)

q_rotated = torch.zeros_like(x)
k_rotated = torch.zeros_like(x)
v_rotated = torch.zeros_like(x)

for position in range(config['context_window']):
    q_rotated[:,position,:] = torch.matmul(q[:,position,:], layer.R[position,:,:])
    k_rotated[:,position,:] = torch.matmul(k[:,position,:], layer.R[position,:,:])
    v_rotated[:,position,:] = torch.matmul(v[:,position,:], layer.R[position,:,:])

q_rotated = (torch.bmm(q.transpose(0,1), layer.R)).transpose(0,1)
k_rotated = (torch.bmm(k.transpose(0,1), layer.R)).transpose(0,1)
v_out = (torch.bmm(v.transpose(0,1), layer.R)).transpose(0,1)

assert torch.allclose(q.transpose(0,1)[0], q[:,0,:])
assert torch.allclose(q.transpose(0,1)[0] @ layer.R[0], q[:,0,:] @ layer.R[0])
assert torch.allclose(q_rotated, q_rotated)

config = {
    'batch_size': 1,
    'd_model': 2,
    'n_heads': 8,
    'context_window': 3,
}

print(Colors.OKGREEN + "### 'config' RoPEAttentionHead 02 ###" + Colors.ENDC)
print(Colors.OKGREEN + str(config) + Colors.ENDC)  
print(Colors.OKGREEN + "###" + Colors.ENDC)
print("")

layer = RoPEAttentionHead(config)
batch = torch.ones((config['batch_size'], config['context_window'], config['d_model']))
output, attn_weights = layer(batch, return_attn_weights=True)

m = 0
x_q = batch[0, m]
q = layer.R[m,:,:] @ layer.w_q(x_q)

assert torch.allclose(layer.w_q(x_q), layer.w_q.weight @ x_q)
assert torch.allclose(q, layer.R[m, :, :] @ layer.w_q.weight @ x_q)

n = 2
x_k = batch[0, n]
k = layer.R[n,:,:] @ layer.w_k(x_k)

assert torch.allclose(layer.w_k(x_k), layer.w_k.weight @ x_k)
assert torch.allclose(k, layer.R[n, :, :] @ layer.w_k.weight @ x_k)

assert q.T @ k == q @ k # transpose is redundant
assert torch.allclose(q @ k, x_k.T @ layer.w_k.weight.T @ layer.R[n, :, :].T @ layer.R[m, :, :] @ layer.w_q.weight @ x_q)
assert torch.allclose(q @ k, x_k.T @ layer.w_k.weight.T @ layer.R[n-m, :, :].T @ layer.w_q.weight @ x_q)


# definitely there's an optimization we could make where we cache the rotation matrices, but skip.
class RoPEMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        print(Colors.WARNING + "Running RoPEMultiheadAttention" + Colors.ENDC)

        self.heads = nn.ModuleList([
            RoPEAttentionHead(config) for _ in range(config['n_heads'])
        ])
        self.linear = nn.Linear(config['n_heads'] * config['d_model'], config['d_model'])
        self.dropout = nn.Dropout(.1)

    def forward(self, x):
        heads = [h(x) for h in self.heads]
        x = torch.cat(heads, dim=-1)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
MASTER_CONFIG.update({
    'n_heads': 8,
})

print(Colors.OKGREEN + "### 'MASTER_CONFIG' RoPEAttentionHead 01 ###" + Colors.ENDC)
print(Colors.OKGREEN + str(MASTER_CONFIG) + Colors.ENDC)  
print(Colors.OKGREEN + "###" + Colors.ENDC)
print("")

layer = RoPEMultiheadAttention(MASTER_CONFIG)
batch = torch.ones((MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'], MASTER_CONFIG['d_model']))
output = layer(batch)
output.shape

class RopeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        print(Colors.WARNING + "Running RopeModel #1" + Colors.ENDC)

        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.rms = RMSNorm((config['context_window'], config['d_model']))
        self.rope_attention = RoPEMultiheadAttention(config)

        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
        )

        self.last_linear = nn.Linear(config['d_model'], config['vocab_size'])

        print(Colors.RED + Colors.BOLD)
        print("model params from RopeModel #1:", sum([m.numel() for m in self.parameters()]))
        print(Colors.ENDC)

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        
        # one block of attention
        x = self.rms(x) # rms pre-normalization
        x = x + self.rope_attention(x)

        x = self.rms(x) # rms pre-normalization
        x = x + self.linear(x)

        logits = self.last_linear(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

        else:
            return logits

print(Colors.OKGREEN + "### 'MASTER_CONFIG' RoPEModel 01 ###" + Colors.ENDC)
print(Colors.OKGREEN + str(MASTER_CONFIG) + Colors.ENDC)  
print(Colors.OKGREEN + "###" + Colors.ENDC)
print("")

model = RopeModel(MASTER_CONFIG)
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

logits, loss = model(xs, ys)
optimizer = torch.optim.Adam(model.parameters(),MASTER_CONFIG['opt_adam_lr'])
train(model, optimizer)

generate(model, config=MASTER_CONFIG)

MASTER_CONFIG.update({
    'n_heads': 8,
})

print(Colors.OKGREEN + "### 'MASTER_CONFIG' RoPEAttentionHead 02 ###" + Colors.ENDC)
print(Colors.OKGREEN + str(MASTER_CONFIG) + Colors.ENDC)  
print(Colors.OKGREEN + "###" + Colors.ENDC)
print("")

layer = RoPEAttentionHead(MASTER_CONFIG)
batch = torch.ones((MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'], MASTER_CONFIG['d_model']))
output, attn_weights = layer(batch, return_attn_weights=True)

plt.imshow(attn_weights[0].detach().numpy(), interpolation='nearest')
plt.colorbar()

config = {
    'batch_size': 24,
    'd_model': 128,
    'n_heads': 8,
    'context_window': 32,
}

class RoPEMaskedAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        print(Colors.WARNING + "Running RoPEMaskedAttentionHead" + Colors.ENDC)

        self.w_q = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.w_k = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.w_v = nn.Linear(config['d_model'], config['d_model'], bias=False)

        self.R = get_rotary_matrix(config['context_window'], config['d_model'])

    def get_rotary_matrix(context_window, embedding_dim):
        R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
        for position in range(context_window):
            for i in range(embedding_dim//2):
                theta = 10000. ** (-2.*(i - 1) / embedding_dim)
                m_theta = position * theta
                R[position, 2*i,2*i] = np.cos(m_theta)
                R[position, 2*i,2*i+1] = - np.sin(m_theta)
                R[position, 2*i+1,2*i] = np.sin(m_theta)
                R[position, 2*i+1,2*i+1] = np.cos(m_theta)
        return R
    
    def forward(self, x, return_attn_weights=False):
        b,m,d = x.shape
        
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q_rotated = (torch.bmm(q.transpose(0,1), self.R[:m])).transpose(0,1)
        k_rotated = (torch.bmm(k.transpose(0,1), self.R[:m])).transpose(0,1)

        activations = F.scaled_dot_product_attention(
            q_rotated,k_rotated,v,dropout_p =.1, is_causal=True
        )

        if return_attn_weights:
            attn_mask = torch.tril(torch.ones((m,m)), diagonal=0)
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1,2)) / np.sqrt(d) + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            return activations, attn_weights
        return activations

print(Colors.OKGREEN + "### 'config' RoPEMaskedAttentionHead 01 ###" + Colors.ENDC)
print(Colors.OKGREEN + str(config) + Colors.ENDC)  
print(Colors.OKGREEN + "###" + Colors.ENDC)
print("")

layer = RoPEMaskedAttentionHead(config)
batch = torch.randn((config['batch_size'], config['context_window'], config['d_model']))
output, attn_weights = layer(batch, return_attn_weights=True)

print(Colors.OKGREEN + "### 'MASTER_CONFIG' RoPEMaskedAttentionHead 02 ###" + Colors.ENDC)
print(Colors.OKGREEN + str(MASTER_CONFIG) + Colors.ENDC)  
print(Colors.OKGREEN + "###" + Colors.ENDC)
print("")

layer = RoPEMaskedAttentionHead(MASTER_CONFIG)
batch = torch.ones((MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'], MASTER_CONFIG['d_model']))
output, attn_weights = layer(batch, return_attn_weights=True)

plt.imshow(attn_weights[0].detach().numpy())
plt.colorbar()

# definitely there's an optimization we could make where we cache the rotation matrices, but skip.
class RoPEMaskedMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        print(Colors.WARNING + "Running RoPEMaskedMultiheadAttention" + Colors.ENDC)

        self.heads = nn.ModuleList([
            RoPEMaskedAttentionHead(config) for _ in range(config['n_heads'])
        ])
        self.linear = nn.Linear(config['n_heads'] * config['d_model'], config['d_model'])
        self.dropout = nn.Dropout(.1)

    def forward(self, x):
        heads = [h(x) for h in self.heads]
        x = torch.cat(heads, dim=-1)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
MASTER_CONFIG.update({
    'n_heads': 8,
})

print(Colors.OKGREEN + "### 'MASTER_CONFIG' RoPEMultiheadAttention 01 ###" + Colors.ENDC)
print(Colors.OKGREEN + str(MASTER_CONFIG) + Colors.ENDC)  
print(Colors.OKGREEN + "###" + Colors.ENDC)
print("")

layer = RoPEMultiheadAttention(MASTER_CONFIG)
batch = torch.ones((MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'], MASTER_CONFIG['d_model']))
output = layer(batch)
output.shape

class RopeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        print(Colors.WARNING + "Running RopeModel #2" + Colors.ENDC)

        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.rms = RMSNorm((config['context_window'], config['d_model']))
        self.rope_attention = RoPEMaskedMultiheadAttention(config)

        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
        )

        self.last_linear = nn.Linear(config['d_model'], config['vocab_size'])
        
        print(Colors.RED + Colors.BOLD)
        print("model params from RopeModel #2:", sum([m.numel() for m in self.parameters()]))
        print(Colors.ENDC)

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        
        # one block of attention
        x = self.rms(x) # rms pre-normalization
        x = x + self.rope_attention(x)

        x = self.rms(x) # rms pre-normalization
        x = x + self.linear(x)

        logits = self.last_linear(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

        else:
            return logits
        
model = RopeModel(MASTER_CONFIG)
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

logits, loss = model(xs, ys)
optimizer = torch.optim.Adam(model.parameters(),MASTER_CONFIG['opt_adam_lr'])
train(model, optimizer)

MASTER_CONFIG.update({
    'epochs': 1000,
    'log_interval': 10,
})

print(Colors.OKGREEN + "### 'MASTER_CONFIG' RopeModel 02 ###" + Colors.ENDC)
print(Colors.OKGREEN + str(MASTER_CONFIG) + Colors.ENDC)  
print(Colors.OKGREEN + "###" + Colors.ENDC)
print("")

train(model, optimizer)

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit
    https://arxiv.org/pdf/2002.05202v1.pdf
    """
    def __init__(self, size):
        super().__init__()
        self.config = config

        print(Colors.WARNING + "Running SwiGLU" + Colors.ENDC)

        self.linear_gate = nn.Linear(size, size)
        self.linear = nn.Linear(size, size)
        self.beta = torch.randn(1, requires_grad=True)

        self.beta = nn.Parameter(torch.ones(1))
        self.register_parameter("beta", self.beta)

    def forward(self, x): 
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        out = swish_gate * self.linear(x)
        return out

class RopeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        print(Colors.WARNING + "Running RopeModel #3" + Colors.ENDC)

        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.rms = RMSNorm((config['context_window'], config['d_model']))
        self.rope_attention = RoPEMaskedMultiheadAttention(config)

        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
        )

        self.last_linear = nn.Linear(config['d_model'], config['vocab_size'])
        
        print(Colors.RED + Colors.BOLD)
        print("model params from RopeModel #3:", sum([m.numel() for m in self.parameters()]))
        print(Colors.ENDC)

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        
        # one block of attention
        x = self.rms(x) # rms pre-normalization
        x = x + self.rope_attention(x)

        x = self.rms(x) # rms pre-normalization
        x = x + self.linear(x)

        logits = self.last_linear(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

        else:
            return logits

model = RopeModel(MASTER_CONFIG)
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

logits, loss = model(xs, ys)
optimizer = torch.optim.Adam(model.parameters(),MASTER_CONFIG['opt_adam_lr'])
train(model, optimizer)

# add RMSNorm and residual conncection
class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        print(Colors.WARNING + "Running LlamaBlock" + Colors.ENDC)

        self.rms = RMSNorm((config['context_window'], config['d_model']))
        
        self.attention = RoPEMaskedMultiheadAttention(config)
        self.feedforward = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
        )

    def forward(self, x):
        x = self.rms(x) # rms pre-normalization
        x = x + self.attention(x)

        x = self.rms(x) # rms pre-normalization
        x = x + self.feedforward(x)
        return x

print(Colors.OKGREEN + "### 'MASTER_CONFIG' LlamaBlock 01 ###" + Colors.ENDC)
print(Colors.OKGREEN + str(MASTER_CONFIG) + Colors.ENDC)  
print(Colors.OKGREEN + "###" + Colors.ENDC)
print("")

block = LlamaBlock(MASTER_CONFIG)
block(torch.randn(MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'], MASTER_CONFIG['d_model']));

from collections import OrderedDict

MASTER_CONFIG.update({
    'n_layers': 8,
})

class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        print(Colors.WARNING + "Running Llama" + Colors.ENDC)

        self.embeddings = nn.Embedding(config['vocab_size'], config['d_model'])
        self.llama_blocks = nn.Sequential(
            OrderedDict([(f"llama_{i}", LlamaBlock(config)) for i in range(config['n_layers'])])
        )

        self.ffn = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

        print(Colors.RED + Colors.BOLD)
        print("model params from Llama:", sum([m.numel() for m in self.parameters()]))
        print(Colors.ENDC)

    def forward(self, idx, targets=None):
        x = self.embeddings(idx)
        x = self.llama_blocks(x)
        logits = self.ffn(x)

        if targets is None:
            return logits
        
        else:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss
        
    def save_model(self, file_path):
        """
        Save the model to a file.

        Args:
            file_path (str): File path to save the model.
        """
        torch.save(self.state_dict(), file_path)
        print(Colors.BOLD + f"Model saved to {file_path}" + Colors.ENDC)
    

print(Colors.OKGREEN + "### 'MASTER_CONFIG' Llama 01 ###" + Colors.ENDC)
print(Colors.OKGREEN + str(MASTER_CONFIG) + Colors.ENDC)  
print(Colors.OKGREEN + "###" + Colors.ENDC)
print("")

llama = Llama(MASTER_CONFIG)
optimizer = torch.optim.Adam(llama.parameters(),MASTER_CONFIG['opt_adam_lr'])
train(llama, optimizer, config=MASTER_CONFIG)

MASTER_CONFIG.update({
    'epochs': 5000,
})

print(Colors.OKGREEN + "### 'MASTER_CONFIG' Llama Train 01 ###" + Colors.ENDC)
print(Colors.OKGREEN + str(MASTER_CONFIG) + Colors.ENDC)  
print(Colors.OKGREEN + "###" + Colors.ENDC)
print("")

train(llama, optimizer, scheduler=None, config=MASTER_CONFIG)

MASTER_CONFIG.update({
    'n_layers': 8,
    'd_model': 128,
    'context_window': 32,
    'batch_size': 24,
    'epochs': 10000,
    'n_heads': 8,
})

print(Colors.OKGREEN + "### 'MASTER_CONFIG' Llama Train Final ###" + Colors.ENDC)
print(Colors.OKGREEN + str(MASTER_CONFIG) + Colors.ENDC)  
print(Colors.OKGREEN + "###" + Colors.ENDC)
print("")

train(llama, optimizer, config=MASTER_CONFIG)

train(llama, optimizer, config=MASTER_CONFIG)

llama.save_model("llama_model.pth")

print(generate(llama, MASTER_CONFIG, 400)[0])
print("")

xs, ys = get_batches(dataset, 'test', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

logits, loss = llama(xs, ys)

print(loss)

# print the percentage that are near 0
def show_grads(model, tol=1e-2):
    return sorted([(name, 100.0 * float(torch.sum(torch.abs(param) <= tol)) / float(param.nelement())) for name, param in model.named_parameters() if param.requires_grad], key=lambda t: t[1], reverse=True)

show_grads(llama)

print(Colors.BOLD + Colors.OKGREEN + "Final config to use in inference: ", MASTER_CONFIG)
print(Colors.ENDC)

print(Colors.BOLD + f"Model Training Started at: {current_date}")
print(f"Model Training Ended at: {datetime.now()}" + Colors.ENDC)
