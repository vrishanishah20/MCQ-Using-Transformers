import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import json

from your_model_file import TransformerGPT2  # Replace with your actual model module

# === 1. Dataset for OpenBookQA in generative format ===
class OpenBookQAGenDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, "r") as f:
            for line in f:
                item = json.loads(line)
                fact = item["fact1"]
                stem = item["question"]["stem"]
                choices = item["question"]["choices"]
                answer = item["answerKey"]

                # Build the prompt
                prompt = f"[START] {fact} {stem} "
                for choice in choices:
                    prompt += f"[{choice['label']}] {choice['text']} "
                prompt += "[ANSWER] "

                # Append correct label to use as target
                full_input = prompt + answer
                tokenized = tokenizer(full_input, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
                self.samples.append({
                    "input_ids": tokenized["input_ids"].squeeze(0),
                    "attention_mask": tokenized["attention_mask"].squeeze(0),
                    "label_id": tokenizer.convert_tokens_to_ids(answer)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# === 2. Load model from pretrained checkpoint ===
def load_model(checkpoint_path, device):
    model = TransformerGPT2()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"✅ Loaded pretrained model from: {checkpoint_path}")
    return model


# === 3. Fine-tune on OpenBookQA ===
def fine_tune_model(model, dataloader, tokenizer, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()

    for epoch in range(3):
        total_loss = 0
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Predict the last token (the answer A/B/C/D)
            loss = F.cross_entropy(logits[:, -1, :], batch["label_id"].to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Average Loss: {total_loss / len(dataloader):.4f}")

    return model


# === 4. Main Script ===
def main():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = OpenBookQAGenDataset("train_complete.jsonl", tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = load_model("pretrained_gpt2.pth", device)
    model = fine_tune_model(model, dataloader, tokenizer, device)

    torch.save(model.state_dict(), "openbookqa_finetuned.pth")
    print("✅ Fine-tuned model saved to openbookqa_finetuned.pth")


if __name__ == "__main__":
    main()
