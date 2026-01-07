import torch # PyTorch
import torch.nn as nn # Neural-Nets
from torch.utils.data import DataLoader, random_split, ConcatDataset # Batching/splitting
from sklearn.metrics import f1_score, precision_score, recall_score # Validation
from datasets import load_dataset # From HuggingFace
from transformers import AutoTokenizer, DistilBertForSequenceClassification # BERT!

MODEL_NAME = "distilbert-base-uncased" # University Professor model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0
bad_dataset = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train")
good_dataset = load_dataset("wikitext", "wikitext-2-v1", split="train[:4000]")

class SafeWrapper(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = [item for item in data if item['text'].strip()]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return {"prompt": self.data[idx]["text"], "violated_categories": ""}

safe_dataset_wrapped = SafeWrapper(good_dataset)

all_labels = set()
for item in bad_dataset:
    labels = [l.strip() for l in item["violated_categories"].split(",")]
    all_labels.update(labels)

all_labels = sorted(list(all_labels))
all_labels = ['SFW' if l == '' else l for l in all_labels]
label2idx = {l:i for i,l in enumerate(all_labels)}
num_labels = len(all_labels)
print(f"Labels (n={num_labels}) = " + ','.join(all_labels))

def encode_labels(categories: str):
    vector = torch.zeros(num_labels)
    if not categories.strip():
        if "SFW" in label2idx: vector[label2idx["SFW"]] = 1
        return vector
    for cat in categories.split(","):
        cat = cat.strip()
        if cat in label2idx: vector[label2idx[cat]] = 1
    return vector

class SafetyDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_len=64):
        self.data=data
        self.max_len=max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["prompt"]

        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        labels = encode_labels(item["violated_categories"])
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

combined_dataset = ConcatDataset([bad_dataset, safe_dataset_wrapped])
total_len = len(combined_dataset)
train_len = int(0.8 * total_len)
val_len = int(0.1 * total_len)
test_len = total_len - train_len - val_len
train_data, val_data, test_data = random_split(combined_dataset, [train_len, val_len, test_len])

train_dataset = SafetyDataset(train_data)
val_dataset = SafetyDataset(val_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Training using " + device)
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="multi_label_classification"
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()

label_thresholds = {k: 0.5 for k in label2idx.keys()} # Assume 0.5 for all else
label_thresholds["Sexual (minor)"] = 0.35 # Lower threshold for safety
label_thresholds["Violence"] = 0.40 # Lower threshold for safety

def evaluate(loader: DataLoader):
    model.eval()
    all_preds, all_labels = [],[]
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=mask)
            logits = outputs.logits

            probs = torch.sigmoid(logits)
            preds = torch.zeros_like(probs).int()
            for i,label in enumerate(label2idx.keys()):
                preds[:, i] = (probs[:, i] > label_thresholds[label]).int()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return f1_score(all_labels, all_preds, average='macro', zero_division=0)

print("Starting Training (this will be slower)...")
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss=0
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=mask)
        logits = outputs.logits

        loss = criterion(logits, labels.float())
        loss.backward() # Back-prop
        optimizer.step()
        total_loss += loss.item()

    val_f1 = evaluate(val_loader)
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, F1 Val: {val_f1:.4f}")

def classify_prompt(prompt: str):
    model.eval()
    encoding = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64, padding="max_length")
    input_ids = encoding['input_ids'].to(device)
    mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=mask).logits
        probs = torch.sigmoid(logits).cpu()
        preds = torch.zeros_like(probs).cpu()
        for i,label in enumerate(label2idx.keys()):
            preds[:, i] = (probs[:, i] > label_thresholds[label]).int()
    
    return [label for label,i in label2idx.items() if preds[0, i] == 1]

print("Ready! Test input (X=exit):")
while (p := input()) != 'X':
    print(classify_prompt(p))
    print("Test another input (X=exit):")


print("Saving model to './my_safety_model'...")
model.save_pretrained("./my_safety_model")
tokenizer.save_pretrained("./my_safety_tokenizer")
print("Model saved! You can load it later without training.")