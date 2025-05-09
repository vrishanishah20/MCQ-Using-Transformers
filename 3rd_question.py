import argparse
import os
import random
import numpy as np
import time
import math
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from transformers import GPT2TokenizerFast

# ==== Helper Functions ====

def read_corpus(filename, tokenizer):
    print(f"\nReading and tokenizing: {filename}...")
    seq = []
    with open(filename, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            tokens = tokenizer(line)
            for t in tokens['input_ids']:
                seq.append(t)
    print(f"Finished reading {filename} | Total tokens: {len(seq)}")
    return seq

def create_nopeak_mask(size):
    return torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.uint8) == 0

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# ==== Model Classes ====

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x.int())

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=4096, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(x.size(-1))
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        x = x + pe
        return self.dropout(x)

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k, q, v = k.transpose(1,2), q.transpose(1,2), v.transpose(1,2)
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.h * self.d_k)
        output = self.out(concat)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        return self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff = FeedForward(d_model, d_ff=2048, dropout=dropout)  # ✅ Fixed here
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.attn(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x

class DecoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class TransformerGPT2(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.decoder = DecoderOnly(vocab_size, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src, mask):
        x = self.decoder(src, mask)
        return self.out(x)

class TokenDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = torch.tensor(self.data[start:end])
        y = torch.tensor(self.data[start+1:end+1])
        return x, y

def get_model(opt, vocab_size):
    model = TransformerGPT2(vocab_size, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    model.to(opt.device)
    return model

# ==== Training and Testing ====

def train_model(model, opt, train_data, valid_data):
    print("\nStarting training...")
    os.makedirs('3rd_question', exist_ok=True)

    train_loader = DataLoader(TokenDataset(train_data, opt.seqlen), batch_size=opt.batchsize, shuffle=True)
    valid_loader = DataLoader(TokenDataset(valid_data, opt.seqlen), batch_size=opt.batchsize)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    train_losses = []
    val_losses = []

    for epoch in range(opt.epochs):
        print(f"\nEpoch {epoch+1}/{opt.epochs}")
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(opt.device), y.to(opt.device)
            optimizer.zero_grad()
            mask = create_nopeak_mask(opt.seqlen).to(opt.device)
            preds = model(x, mask)
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 500 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(opt.device), y.to(opt.device)
                mask = create_nopeak_mask(opt.seqlen).to(opt.device)
                preds = model(x, mask)
                loss = F.cross_entropy(preds.view(-1, preds.size(-1)), y.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(valid_loader)
        val_losses.append(avg_val_loss)

        print(f"Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

    print("\nSaving logs and model...")
    np.savetxt('3rd_question/train_loss.txt', np.array(train_losses))
    np.savetxt('3rd_question/val_loss.txt', np.array(val_losses))

    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.legend()
    plt.title('Training and Validation Loss (3rd Question)')
    plt.savefig('3rd_question/loss_curve.png')
    plt.close()

    torch.save(model.state_dict(), '3rd_question/model_final.pt')
    print("Training completed and files saved!\n")

def test_model(model, opt, test_data):
    print("Starting testing...")
    model.eval()
    test_loader = DataLoader(TokenDataset(test_data, opt.seqlen), batch_size=opt.batchsize)
    total_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(opt.device), y.to(opt.device)
            mask = create_nopeak_mask(opt.seqlen).to(opt.device)
            preds = model(x, mask)
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), y.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    perplexity = math.exp(avg_loss)
    print(f"Test Perplexity: {perplexity:.2f}")

def main():
    print("\nParsing arguments and setting up...")
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, default=1)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-batchsize', type=int, default=2)
    parser.add_argument('-lr', type=float, default=5e-4)
    parser.add_argument('-seqlen', type=int, default=512)
    opt = parser.parse_args(args=[])

    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {opt.device}")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    train_data = read_corpus('wiki103.train.txt', tokenizer)
    valid_data = read_corpus('wiki103.valid.txt', tokenizer)
    test_data = read_corpus('wiki103.test.txt', tokenizer)

    model = get_model(opt, vocab_size=50257)

    start_time = time.time()
    train_model(model, opt, train_data, valid_data)
    end_time = time.time()

    elapsed_time = end_time - start_time
    total_tokens = len(train_data)
    tokens_per_sec = total_tokens / elapsed_time

    print(f"\nTraining time: {elapsed_time:.2f} seconds")
    print(f"Tokens per second: {tokens_per_sec:.2f}")

    model.load_state_dict(torch.load('3rd_question/model_final.pt'))
    test_model(model, opt, test_data)

if __name__ == "__main__":
    main()