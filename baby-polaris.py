import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Setting the number of threads for PyTorch
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

print(f"{Colors.HEADER}Training Start{Colors.ENDC}")

with open('input.txt', 'r') as file:
    lines = file.read()
print(f"{Colors.OKBLUE}Input text loaded. Length: {len(lines)} characters.{Colors.ENDC}")

vocab = sorted(list(set(lines)))
print(f"{Colors.OKBLUE}Vocabulary constructed. Size: {len(vocab)}{Colors.ENDC}")

itos = {i: ch for i, ch in enumerate(vocab)}
stoi = {ch: i for i, ch in enumerate(vocab)}

def encode(s):
    return [stoi[ch] for ch in s if ch in stoi]

class TextDataset(Dataset):
    def __init__(self, text, sequence_length=10):
        self.encoded_text = encode(text)
        self.seq_length = sequence_length
        print(f"{Colors.OKBLUE}TextDataset initialized. Sequence length: {sequence_length}, Encoded text length: {len(self.encoded_text)}{Colors.ENDC}")

    def __len__(self):
        return len(self.encoded_text) - self.seq_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.encoded_text[index:index+self.seq_length]),
            torch.tensor(self.encoded_text[index+1:index+self.seq_length+1])
        )

class SmallRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        print(f"{Colors.OKBLUE}SmallRNNModel initialized. Embedding dim: {embed_dim}, Hidden dim: {hidden_dim}, Vocab size: {vocab_size}{Colors.ENDC}")

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

vocab_size = len(vocab)
embed_dim = 64
hidden_dim = 128
learning_rate = 0.005
epochs = 1 
batch_size = 32

model = SmallRNNModel(vocab_size, embed_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

print(f"{Colors.OKBLUE}Model, criterion, and optimizer initialized.{Colors.ENDC}")

dataset = TextDataset(lines, sequence_length=64)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
print(f"{Colors.OKBLUE}Dataloader prepared. Batch size: {batch_size}{Colors.ENDC}")  # Corrected batch size in print statement

total_steps = len(dataloader) * epochs  # Correct total steps calculation
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps)
print(f"{Colors.OKBLUE}Scheduler configured. Total steps: {total_steps}, Max LR: {learning_rate}{Colors.ENDC}")

loss_threshold = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.transpose(1, 2), targets)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:  # Adjust the interval to your preference
            current_lr = scheduler.get_last_lr()[0]
            print(f"{Colors.BOLD}Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}, LR: {current_lr:.6f}{Colors.ENDC}")

    avg_loss = total_loss / len(dataloader)
    print(f"{Colors.OKBLUE}Epoch [{epoch+1}/{epochs}] completed, Average Loss: {avg_loss:.4f}{Colors.ENDC}")

# Save model, optimizer, and scheduler states
model_path = "small_rnn_model_checkpoint.pth"
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss.item(),
}, model_path)
print(f"{Colors.BOLD}{Colors.OKGREEN}Checkpoint saved at {model_path}{Colors.ENDC}")

if loss.item() >= loss_threshold:
    print(f"{Colors.OKGREEN}Completed all epochs without reaching the loss threshold.{Colors.ENDC}")

print(f"{Colors.OKGREEN}Training Completed{Colors.ENDC}")

model_path = "small_rnn_model.pth"
torch.save(model.state_dict(), model_path)
print(f"{Colors.BOLD}{Colors.OKGREEN}Model saved successfully at {model_path}{Colors.ENDC}")

model.load_state_dict(torch.load(model_path))
model.eval()
print(f"{Colors.BOLD}{Colors.OKBLUE}Model loaded successfully for inference.{Colors.ENDC}")

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
       Args:
           logits: logits distribution shape (vocabulary size)
           top_k > 0: keep only top k tokens with highest probability (top-k filtering).
           top_p > 0.0: keep the top tokens with a cumulative probability >= top_p (nucleus filtering).
           filter_value: The value to assign to filtered tokens, default to -float('Inf')
       Returns:
           logits: the filtered logits distribution of shape (vocabulary size)
    """
    assert logits.dim() == 1  # batch dimension 1 for single sample

    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def generate_text(seed_text, model, max_length=100, temperature=0.8, top_k=30, top_p=0.7):
    model.eval()  # Ensure the model is in evaluation mode
    text_generated = [seed_text]
    input_eval = torch.tensor(encode(seed_text), dtype=torch.long).unsqueeze(0)  # Prepare input

    with torch.no_grad():  # Inference without tracking gradients
        for _ in range(max_length):
            predictions = model(input_eval)[:,-1,:]  # Predict the next token
            predictions = predictions / temperature  # Apply temperature
            filtered_logits = top_k_top_p_filtering(predictions.squeeze(), top_k=top_k, top_p=top_p)
            probs = F.softmax(filtered_logits, dim=-1)
            predicted_id = torch.multinomial(probs, num_samples=1).item()

            if itos[predicted_id] == '<END>':  # Assuming '<END>' is your end-of-sequence token
                print("\n")  # Move to a new line after finishing the generation
                break

            generated_character = itos[predicted_id]
            print(generated_character, end='', flush=True)  # Print in real time without adding a new line

            # Prepare the next input batch
            predicted_id_tensor = torch.tensor([[predicted_id]], dtype=torch.long)
            input_eval = torch.cat([input_eval, predicted_id_tensor], dim=1)

            # Append the generated character for return value
            text_generated.append(generated_character)

    return ''.join(text_generated)

seed_text = "O que é deus? "
