import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from datetime import datetime
import argparse

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#
# Start
#

current_date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

print(Colors.BOLD)
print(f"Polaris LLM Inferencer v0.1.0")
print(f"Inference started: {current_date}")

#
# Model Components
#

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit
    https://arxiv.org/pdf/2002.05202v1.pdf
    Implements the SwiGLU activation function.
    """
    def __init__(self, size):
        super().__init__()
        self.config = config
        self.linear_gate = nn.Linear(size, size)
        self.linear = nn.Linear(size, size)
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x): 
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        return swish_gate * self.linear(x)

def get_rotary_matrix(context_window, embedding_dim):
    """
    Generate a rotary positional encoding matrix.
    :param context_window: The size of the context window.
    :param embedding_dim: The dimension of the embeddings.
    :return: A rotary positional encoding matrix.
    """
    R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
    for position in range(context_window):
        for i in range(embedding_dim // 2):
            theta = 10000. ** (-2. * (i - 1) / embedding_dim)
            m_theta = position * theta
            R[position, 2 * i, 2 * i] = np.cos(m_theta)
            R[position, 2 * i, 2 * i + 1] = -np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i] = np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i + 1] = np.cos(m_theta)
    return R

class RoPEMaskedAttentionHead(nn.Module):
    """
    Rotary Positional Encoding Masked Attention Head.
    Implements an attention head with RoPE.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.w_q = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.w_k = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.w_v = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.R = get_rotary_matrix(config['context_window'], config['d_model'])

    def forward(self, x, return_attn_weights=False):
        b, m, d = x.shape
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        q_rotated = (torch.bmm(q.transpose(0, 1), self.R[:m])).transpose(0, 1)
        k_rotated = (torch.bmm(k.transpose(0, 1), self.R[:m])).transpose(0, 1)
        activations = F.scaled_dot_product_attention(q_rotated, k_rotated, v, dropout_p=0.1, is_causal=False)
        if return_attn_weights:
            attn_mask = torch.tril(torch.ones((m, m)), diagonal=0)
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1, 2)) / np.sqrt(d) + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            return activations, attn_weights
        return activations

class RoPEMaskedMultiheadAttention(nn.Module):
    """
    Rotary Positional Encoding Masked Multihead Attention.
    Implements multihead attention with RoPE.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList([RoPEMaskedAttentionHead(config) for _ in range(config['n_heads'])])
        self.linear = nn.Linear(config['n_heads'] * config['d_model'], config['d_model'])
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        heads = [h(x) for h in self.heads]
        x = torch.cat(heads, dim=-1)
        x = self.linear(x)
        return self.dropout(x)

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    def __init__(self, layer_shape, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(layer_shape))

    def forward(self, x):
        ff_rms = torch.linalg.norm(x, dim=(1, 2)) * x[0].numel() ** -0.5
        return self.scale[:x.shape[1], :].unsqueeze(0) * (x / ff_rms.unsqueeze(-1).unsqueeze(-1))

class LlamaBlock(nn.Module):
    """
    Llama Transformer Block.
    A single transformer block in Llama model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rms = RMSNorm((config['context_window'], config['d_model']))
        self.attention = RoPEMaskedMultiheadAttention(config)
        self.feedforward = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
        )

    def forward(self, x):
        x = self.rms(x)  # RMS pre-normalization
        x = x + self.attention(x)
        x = self.rms(x)  # RMS pre-normalization
        return x + self.feedforward(x)

class Llama(nn.Module):
    """
    Llama Language Model.
    Implements the full Llama language model architecture.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config['vocab_size'], config['d_model'])
        self.llama_blocks = nn.Sequential(
            OrderedDict([(f"llama_{i}", LlamaBlock(config)) for i in range(config['n_layers'])])
        )
        self.ffn = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

    def forward(self, idx, targets=None):
        x = self.embeddings(idx)
        x = self.llama_blocks(x)
        logits = self.ffn(x)
        if targets is None:
            return logits
        else:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

#
# Text generation tools
#

def decode(ids, itos):
    """
    Convert a list of ids to text using the inverse token mapping.
    :param ids: List of token ids.
    :param itos: Inverse token mapping dictionary.
    :return: Decoded string.
    """
    return ''.join([itos[i.item()] for i in ids])

def generate(model, config, max_new_tokens=30, starting_idx=None, temperature=1.0, top_k=0, top_p=1.0, stop_phrase="Mirim"):
    if starting_idx is None:
        idx = torch.zeros(1, config['context_window']).long()
    else:
        idx = starting_idx

    generated_text = ''

    for _ in range(max_new_tokens):
        logits = model(idx)
        last_time_step_logits = logits[:, -1, :] / temperature

        # Apply top_k filtering
        if top_k > 0:
            indices_to_remove = last_time_step_logits < torch.topk(last_time_step_logits, top_k)[0][..., -1, None]
            last_time_step_logits[indices_to_remove] = -float('Inf')

        # Apply top_p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(last_time_step_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            last_time_step_logits[indices_to_remove] = -float('Inf')

        p = F.softmax(last_time_step_logits, dim=-1)
        idx_next = torch.multinomial(p, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=1)[:, -config['context_window']:]
        generated_token = decode(idx_next, itos)

        generated_text += generated_token
        print(generated_token, end='', flush=True)

        # Check if the cumulative text ends with the stop phrase
        if generated_text.endswith(stop_phrase):
            break

    print()
    return generated_text


#
# Main Execution
#

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Polaris LLM Inferencer')
    parser.add_argument('--maxtokens', type=int, default=200, help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=0, help='Top K filtering')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top P (nucleus) filtering')
    parser.add_argument('--user_input', action='store_true', help='Enable user input for text generation')

    
    args = parser.parse_args()
    
    #
    # Load vocabulary
    #
    
    try:
        fileOpen = 'input.txt'
        lines = open(fileOpen, 'r').read()
        print(Colors.BOLD + f"Loading Vocab: {fileOpen}" + Colors.ENDC)
    except FileNotFoundError:
        print(Colors.RED + "Input/Vocab file not found." + Colors.ENDC)
        exit(1)

    vocab = sorted(list(set(lines)))
    tags = ['<START>', '<END>', '<POLARIS>', '<USER>']
    for tag in tags:
        vocab.append(tag)

    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    
    #
    # Configuration
    #
    
    MASTER_CONFIG = {
        'vocab_size': len(vocab),
        'batch_size': 32,
        'context_window': 256,
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 2
    }
    config = MASTER_CONFIG
    
    #
    # Load Model
    #
    
    model_path = "llama_model.pth"
    try:
        print(Colors.BOLD + f"Loading Model: {model_path}")
        print(Colors.ENDC)
        model = Llama(MASTER_CONFIG)
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    
    #
    # Run Text Generation
    #
        
    # Conditional user input based on the flag
    if args.user_input:
        user_input = input("> ")
    else:
        user_input = "<START>"  # Or any default starting string you prefer

    # Tokenize input
    tokenized_input = [stoi[ch] for ch in user_input if ch in stoi]
    tokenized_input = torch.tensor(tokenized_input).unsqueeze(0)  # Add batch dimension

    # Run Text Generation with user input
    generated_text = generate(model, MASTER_CONFIG, args.maxtokens, starting_idx=tokenized_input)

    print(f"\nInference finished: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

