import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
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

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

HP = {
    "embed_dim": 64,
    "hidden_dim": 128,
    "learning_rate": 0.0005,
    "epochs": 1,
    "batch_size": 128,
    "loss_threshold": 1.6,
    "num_layers": 2,
    "sequence_length": 256,
}

print(f"{Colors.HEADER}Training Start{Colors.ENDC}")

with open('input.txt', 'r') as file:
    lines = file.read()
print(f"{Colors.OKBLUE}Input text loaded. Length: {len(lines)} characters.{Colors.ENDC}")

vocab = sorted(list(set(lines)))
print(f"{Colors.OKBLUE}Vocabulary constructed. Size: {len(vocab)}{Colors.ENDC}")

HP.update({"vocab_size": len(vocab)})

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
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout=0.5, num_layers=HP['num_layers']):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        print(f"{Colors.OKBLUE}SmallRNNModel initialized. Embedding dim: {embed_dim}, Hidden dim: {hidden_dim}, Vocab size: {vocab_size}{Colors.ENDC}")

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

model = SmallRNNModel(HP['vocab_size'], HP['embed_dim'], HP['hidden_dim'])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=HP['learning_rate'])

dataset = TextDataset(lines, sequence_length=HP['sequence_length'])
dataloader = DataLoader(dataset, batch_size=HP['batch_size'], shuffle=True, num_workers=8)
print(f"{Colors.OKBLUE}Dataloader prepared. Batch size: {HP['batch_size']}{Colors.ENDC}")

# Checkpoint path
model_path = "small_rnn_model_checkpoint.pth"

# Check if checkpoint exists
if os.path.isfile(model_path):
    print(f"{Colors.OKGREEN}Checkpoint found, loading...{Colors.ENDC}")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']  # Next epoch
    loss = checkpoint['loss']
else:
    start_epoch = 0  # Start from the beginning

total_steps = len(dataloader) * (HP['epochs'] - start_epoch)
scheduler = OneCycleLR(optimizer, max_lr=HP['learning_rate'], total_steps=total_steps)
print(f"{Colors.OKBLUE}Scheduler configured. Total steps: {total_steps}, Max LR: {HP['learning_rate']}{Colors.ENDC}")

for epoch in range(start_epoch, HP['epochs']):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.transpose(1, 2), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if loss.item() < HP['loss_threshold']:
            print(f"{Colors.WARNING}Loss {loss.item():.4f} is below threshold {HP['loss_threshold']:.4f}{Colors.ENDC}")
            break

        if batch_idx % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"{Colors.BOLD}Epoch [{epoch+1}/{HP['epochs']}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}, LR: {current_lr:.6f}{Colors.ENDC}")

    avg_loss = total_loss / len(dataloader)
    print(f"{Colors.OKBLUE}Epoch [{epoch+1}/{HP['epochs']}] completed, Average Loss: {avg_loss:.4f}{Colors.ENDC}")

    # Save checkpoint after each epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss.item(),
    }, model_path)
    print(f"{Colors.BOLD}{Colors.OKGREEN}Checkpoint saved at {model_path}{Colors.ENDC}")

    if avg_loss < HP['loss_threshold']:
        print(f"{Colors.WARNING}Early stopping triggered at epoch {epoch+1} with loss {avg_loss:.4f}, below threshold {HP['loss_threshold']:.4f}{Colors.ENDC}")
        break  # Exit the training loop

if avg_loss >= HP['loss_threshold']:
    print(f"{Colors.OKGREEN}Completed all epochs without reaching the loss threshold.{Colors.ENDC}")

print(f"{Colors.OKGREEN}Training Completed{Colors.ENDC}")

model_path_final = "small_rnn_model.pth"
torch.save(model.state_dict(), model_path_final)
print(f"{Colors.BOLD}{Colors.OKGREEN}Model saved successfully at {model_path_final}{Colors.ENDC}")
