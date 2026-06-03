import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import math
import os
import re

batch_size = 32
block_size = 256 
n_embd = 512
n_layer = 8
n_head = 8
max_epochs = 80
learning_rate = 1e-4
weight_decay = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = 'checkpoint.pt'
best_model_path = 'best_model.pt'

MAX_CHARS = 1_000_000  #Uses only the first 1 million characters to reduce training time

SAVE_EVERY_N_BATCHES = 500
GENERATE_EVERY_N_BATCHES = 1000  

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    def __len__(self):
        return len(self.data) - self.block_size
    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.block_size+1]#Creates an input sequence
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)

        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))# Prevent tokens from looking ahead to future tokens
        att = F.softmax(att, dim=-1)# Convert attention scores into probabilities
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.dropout(y)
        return y

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.sa = MultiHeadSelfAttention(n_embd, n_head, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer, n_head, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        tok_emb = self.token_embedding(idx)# Looks up token embeddings 
        pos_emb = self.position_embedding(pos)# Looks up position embeddings
        x = tok_emb + pos_emb # Combine token meaning with position information
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        space_idx = char2idx[' ']
        for _ in range(max_new_tokens):
            if idx.size(1) == 0:
                raise ValueError("Empty input sequence in generate()")
            last_idx = idx[0, -1].item()
            if last_idx < 0 or last_idx >= self.token_embedding.num_embeddings:
                raise ValueError(f"Invalid index {last_idx} in generate input")
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:# Keep only the k most likely next characters
                values, indices = torch.topk(logits, top_k)
                logits_filtered = torch.full_like(logits, float('-inf'))
                logits_filtered.scatter_(1, indices, values)
            else:
                logits_filtered = logits

            # Keep the smallest set of characters whose cumulative probability exceeds top_p
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits_filtered, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                sorted_logits[sorted_mask] = float('-inf')

                logits_filtered = torch.full_like(logits_filtered, float('-inf'))
                logits_filtered.scatter_(1, sorted_indices, sorted_logits)

            # Avoid invalid probability distributions if filtering removes every option
            all_inf_mask = torch.isneginf(logits_filtered).all(dim=-1)
            if all_inf_mask.any():
                logits_filtered[all_inf_mask] = logits[all_inf_mask]

            probs = F.softmax(logits_filtered, dim=-1)

            # Ensure probabilities can still be normalized
            denom = probs.sum(dim=-1, keepdim=True)
            if torch.any(denom <= 0):
                raise RuntimeError("Top-p removed all probability mass")
            probs = probs / (denom + 1e-12)

            #Safety checks to catch numerical issues during generation
            assert not torch.isnan(logits_filtered).any(), "NaN in logits_filtered"
            assert not torch.isnan(probs).any(), "NaN in probs"
            assert (probs >= 0).all(), "Negative probs"
            assert probs.sum() > 0, "Zero probability mass"

            #Slightly reduce the chance of generating repeated spaces
            if last_idx == space_idx:
                probs[0, space_idx] *= 0.5
                # Renormalize after penalty
                probs = probs / probs.sum(dim=-1, keepdim=True)

            # space_prob = probs[0, space_idx].item()
            # print(f"P(space) = {space_prob:.3f}")

            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

def encode(s):
    return [char2idx[c] for c in s]

def decode(l):
    return ''.join([idx2char[i] for i in l])

def validate():
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            logits_val = model(x_val)
            loss_val = criterion(logits_val.view(-1, vocab_size), y_val.view(-1))
            val_loss += loss_val.item() * x_val.size(0)  # Accumulate loss weighted by batch size

    avg_val_loss = val_loss / len(val_loader.dataset)
    print(f"Validation loss: {avg_val_loss:.4f}")
    return avg_val_loss

def train():
    global best_val_loss
    for epoch in range(start_epoch, max_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            assert x.max().item() < vocab_size, f"x contains invalid index {x.max().item()}"
            assert y.max().item() < vocab_size, f"y contains invalid index {y.max().item()}"

            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            # Finite loss check to catch exploding gradients
            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss became {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  #Prevent extremely large gradients from destabilizing training
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch} Batch {batch_idx} Avg Loss {avg_loss:.4f}")

            if batch_idx > 0 and batch_idx % SAVE_EVERY_N_BATCHES == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'loss': loss.item()
                }, checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch} batch {batch_idx}")

            if GENERATE_EVERY_N_BATCHES and batch_idx > 0 and batch_idx % GENERATE_EVERY_N_BATCHES == 0:
                model.eval()
                prompt = "To be, or not to be"
                with torch.no_grad():
                    idx = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
                    generated_idx = model.generate(idx, max_new_tokens=100, temperature=0.7, top_p=0.9)
                generated_text = decode(generated_idx[0].tolist())
                clean_text = ' '.join(generated_text.split())
                print("Sample generation:")
                print(clean_text)
                model.train()

        val_loss = validate()

        print(f"Epoch {epoch} completed. Train Loss: {total_loss / len(train_loader):.4f} Val Loss: {val_loss:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'loss': val_loss
        }, checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch} with val loss {best_val_loss:.4f}")

        scheduler.step()

def generate_sample(prompt, max_new_tokens=200, temperature=0.7, top_p=0.9):
    model.eval()
    with torch.no_grad():
        idx = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        generated_idx = model.generate(idx, max_new_tokens, temperature=temperature, top_p=top_p)
    generated_text = decode(generated_idx[0].tolist())
    clean_text = ' '.join(generated_text.split())
    print(clean_text)

def main():
    global chars, vocab_size, char2idx, idx2char, data, train_loader, val_loader
    global model, optimizer, scheduler, criterion, start_epoch, best_val_loss

    print("Script started")

    #Load and preprocess data
    with open('input.txt', 'r', encoding='utf-8') as f:
        full_text = f.read()

    text = full_text[:MAX_CHARS]
    # Normalize excessive spaces and blank lines
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    print(f"Using first {len(text)} characters out of {len(full_text)} total after cleaning")

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size}")

    char2idx = {ch:i for i,ch in enumerate(chars)}
    idx2char = {i:ch for i,ch in enumerate(chars)}

    print(f"space_idx = {char2idx.get(' ', None)}, vocab_size = {vocab_size}")

    data = torch.tensor(encode(text), dtype=torch.long)

    # Safety check on data indices
    assert data.max().item() < vocab_size, f"Data contains index {data.max().item()} >= vocab size {vocab_size}"
    assert data.min().item() >= 0, f"Data contains negative index {data.min().item()}"

    # Split the text into separate training and validation sections.
    # This prevents overlapping sequences from appearing in both sets.
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_dataset = CharDataset(train_data, block_size)
    val_dataset = CharDataset(val_data, block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    print(f"Number of training batches per epoch: {len(train_loader)}")

    model = GPT(vocab_size, n_embd, n_layer, n_head, block_size).to(device) # Create the GPT model and move it to CPU or GPU

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(checkpoint_path): # Resume training from the last saved checkpoint if one exists
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        checkpoint_vocab_size = checkpoint['model_state_dict']['token_embedding.weight'].shape[0]
        if checkpoint_vocab_size != vocab_size:
            print("Error: Vocabulary size mismatch between checkpoint and current data.")
            exit(1)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
        print(f"Resuming from epoch {start_epoch}")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--generate', action='store_true', help='Generate text from prompt')
    parser.add_argument('--prompt', type=str, default="To be, or not to be", help='Prompt for generation')
    args = parser.parse_args()

    if args.train:
        train()
    if args.generate:
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print("Loaded best model for generation.")
        else:
            print("Best model not found, using current weights.")
        generate_sample(args.prompt)

if __name__ == "__main__":
    main()