import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from classification import OpenBookQADataset, BERTMCQClassifier, train, evaluate
import torch.optim as optim
import json
from utils import load_data

import os
import glob
import re

def get_latest_checkpoint(checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "bert_mcq_epoch_*.pt"))
    if not checkpoints:
        return None, 0
    latest = max(checkpoints, key=os.path.getctime)
    match = re.search(r"epoch_(\d+)", latest)
    start_epoch = int(match.group(1)) if match else 0
    return latest, start_epoch

def run_mcqa_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_examples = load_data("train_complete.jsonl")
    val_examples = load_data("dev_complete.jsonl")
    test_examples = load_data("test_complete.jsonl")

    train_dataset = OpenBookQADataset(train_examples, tokenizer)
    val_dataset = OpenBookQADataset(val_examples, tokenizer)
    test_dataset = OpenBookQADataset(test_examples, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    model = BERTMCQClassifier().to(device)
    latest_ckpt, start_epoch = get_latest_checkpoint()

    if latest_ckpt:
        model.load_state_dict(torch.load(latest_ckpt, map_location=device))
        print(f"Resumed from checkpoint: {latest_ckpt}")
    else:
        print("Starting from scratch.")
        start_epoch = 0

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(start_epoch, 5):
        train_loss = train(model, train_loader, optimizer, loss_fn, device, epoch)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")


if __name__ == "__main__":
    run_mcqa_classifier()
