import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import json
import os
from utils import load_data  # Import from starter.py

# === 1. Dataset Class ===
class OpenBookQADataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        for item in examples:
            fact = item["fact1"]
            stem = item["question"]["stem"]
            choices = item["question"]["choices"]
            label = item["answerKey"]
            label_idx = ord(label) - ord("A")

            texts = [f"{fact} {stem} {choice['text']}" for choice in choices]
            self.data.append((texts, label_idx))
    
    def __len__(self):
        return len(self.data)
    
    #getting tesnor for each choice text and label, used when model calls each item to compre with actual label
    def __getitem__(self, idx):
        texts, label = self.data[idx]
        encoding = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "label": torch.tensor(label)
        }


# === 2. Model Definition ===
class BERTMCQClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        B, N, L = input_ids.shape
        input_ids = input_ids.view(B * N, L)
        attention_mask = attention_mask.view(B * N, L)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(cls_embeddings)  # shape: (B * N, 1)
        return logits.view(B, N)  # reshape to (B, 4)
    
# === 3. Training Utilities ===
def train(model, dataloader, optimizer, loss_fn, device, epoch, checkpoint_dir="checkpoints"):
    model.train()
    total_loss = 0
    os.makedirs(checkpoint_dir, exist_ok=True)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

    # Save model checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"bert_mcq_epoch_{epoch+1}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    return avg_loss


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total



