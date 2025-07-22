import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import random
import platform
import time

# 🔹 Base directory
ROOT_DIR = "../../autodl-tmp/pkl_Reveal_path_fusion_CFG_withoutVWS"
# Set the selected model names
#model_name = "BiLSTM"  # LSTM, BiLSTM, GRU, BiGRU or CNN
model_names = ["LSTM", "BiLSTM", "GRU", "BiGRU", "CNN"]

# 🔹 Set hyperparameters
MAX_LEN = 200  # Unified sequence length
INPUT_SIZE = 128  # Dimension of each vector
HIDDEN_SIZE = 64
OUTPUT_SIZE = 2  # Binary classification
NUM_LAYERS = 1
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001

# 🔹 Paths for train/validation/test data
TRAIN_DATA_PATH = os.path.join(ROOT_DIR, "train.pkl")
VAL_DATA_PATH = os.path.join(ROOT_DIR, "val.pkl")
TEST_DATA_PATH = os.path.join(ROOT_DIR, "test.pkl")

# 🔹 Directory to save models
MODEL_DIR = os.path.join(ROOT_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)  # ✅ Ensure directory exists
# MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "model.pth")  # ✅ Generate correct file path

# 📌 Dataset class
class PklDataset(Dataset):
    def __init__(self, pkl_file, max_len=200, device="cuda"):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)  # Load pkl file

        # 🔹 Process data structure
        if isinstance(data, dict):
            print(f"📌 Loaded dict format data with {len(data)} folds, concatenating all folds")
            data = pd.concat(data.values(), ignore_index=True)  # Concatenate all folds
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"❌ Data format error, expected DataFrame but got {type(data)}")

        self.max_len = max_len  # Set max sequence length
        self.device = device

        # 🔹 Process 'data' column, ensure consistent format
        self.samples = [self.process_data(row["data"]) for _, row in data.iterrows()]
        self.labels = [torch.tensor(row["label"], dtype=torch.long) for _, row in data.iterrows()]

    def process_data(self, data):
        """ Process data to ensure shape=(200, 128) """
        # ✅ 1. Convert list to numpy array
        tensor_data = torch.tensor(np.array(data), dtype=torch.float32)  # shape (seq_len, 128)
        seq_len = tensor_data.shape[0]

        # ✅ 2. Pad or truncate sequence
        if seq_len >= self.max_len:
            return tensor_data[:self.max_len]  # ✅ Truncate
        else:
            pad_size = self.max_len - seq_len
            return torch.cat([tensor_data, torch.zeros((pad_size, 128), dtype=torch.float32)])  # ✅ Pad zeros

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx].to(self.device), self.labels[idx].to(self.device)

# 📌 collate_fn: simply `stack`
def collate_fn(batch):
    sequences, labels = zip(*batch)
    return torch.stack(sequences).to("cuda"), torch.tensor(labels).to("cuda")

# 📌 LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, device="cuda"):
        super(LSTMModel, self).__init__()
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Take last time step
        return output
    
# 📌 BiLSTM model
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, device="cuda"):
        super(BiLSTMModel, self).__init__()
        self.device = device
        # ✅ BiLSTM: bidirectional=True
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # ✅ Bidirectional doubles hidden dim

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        output = self.fc(lstm_out[:, -1, :])  # ✅ Take output from last time step
        return output

# 📌 GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, device="cuda"):
        super(GRUModel, self).__init__()
        self.device = device
        # ✅ Use GRU instead of LSTM
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)  # Linear layer outputs final classes

    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = self.fc(gru_out[:, -1, :])  # ✅ Take output from last time step
        return output
    
# 📌 BiGRU model
class BiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, device="cuda"):
        super(BiGRUModel, self).__init__()
        self.device = device
        # ✅ Use bidirectional GRU
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Linear layer outputs final classes (bidirectional hidden * 2)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = self.fc(gru_out[:, -1, :])  # ✅ Take output from last time step
        return output

# 📌 CNN model
class CNNModel(nn.Module):
    def __init__(self, input_size, num_filters=64, kernel_sizes=[3, 5, 7], output_size=2, device="cuda"):
        super(CNNModel, self).__init__()
        self.device = device

        # 🔹 Multiple convolution kernels to extract features of different window sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])

        self.fc = nn.Linear(num_filters * len(kernel_sizes), output_size)  # Linear layer outputs final classes

    def forward(self, x):
        x = x.permute(0, 2, 1)  # ✅ Change shape to (batch, channels, seq_len) to fit Conv1d

        conv_outputs = [torch.relu(conv(x)) for conv in self.convs]  # 🔹 Extract features with multiple kernels
        pooled_outputs = [torch.max(out, dim=2)[0] for out in conv_outputs]  # 🔹 Global max pooling
        
        x = torch.cat(pooled_outputs, dim=1)  # 🔹 Concatenate results from different kernels
        output = self.fc(x)  # 🔹 Fully connected layer

        return output

# 📌 Training function
def train(model, train_loader, criterion, optimizer, device="cuda"):
    model.train()
    total_loss, correct = 0, 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    
    return total_loss / len(train_loader), correct / len(train_loader.dataset)

# 📌 Evaluation function
def evaluate(model, val_loader, criterion, device="cuda"):
    model.eval()
    total_loss, correct = 0, 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
    
    return total_loss / len(val_loader), correct / len(val_loader.dataset)

# 📌 Test function: calculate P, R, F1, ACC and return result dict
def test(model, test_loader, device="cuda"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:  # Expecting two values here
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"✅ Test set metrics:")
    print(f"🔹 Precision (P): {precision:.4f}")
    print(f"🔹 Recall (R): {recall:.4f}")
    print(f"🔹 F1 Score (F1): {f1:.4f}")
    print(f"🔹 Accuracy (ACC): {acc:.4f}")
    print("📊 Classification Report:\n", classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1"]))

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# Set random seed
def set_seed(seed=42):
    print(f"🌱 Setting random seed: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Print environment info
def print_env_info():
    print("🧠 PyTorch version:", torch.__version__)
    print("⚙️ CUDA version:", torch.version.cuda)
    print("🖥️ Platform:", platform.platform())
    print("📆 Current time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

def main():
    set_seed(42)
    print_env_info()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Using device: {device.upper()}")

    print("📦 Loading datasets...")
    train_dataset = PklDataset(TRAIN_DATA_PATH, max_len=MAX_LEN, device=device)
    val_dataset = PklDataset(VAL_DATA_PATH, max_len=MAX_LEN, device=device)
    test_dataset = PklDataset(TEST_DATA_PATH, device=device)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

    results_log = []
    results_log.append("Model\tAccuracy\tPrecision\tRecall\tF1\n")

    for model_name in model_names:
        print(f"\n🔥 Training model: {model_name}")

        if model_name == "LSTM":
            model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
        elif model_name == "BiLSTM":
            model = BiLSTMModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
        elif model_name == "GRU":
            model = GRUModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
        elif model_name == "BiGRU":
            model = BiGRUModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
        elif model_name == "CNN":
            model = CNNModel(INPUT_SIZE, num_filters=64, kernel_sizes=[3, 5, 7], output_size=OUTPUT_SIZE).to(device)
        else:
            raise ValueError(f"❌ Unsupported model type: {model_name}")

        MODEL_SAVE_PATH = os.path.join(MODEL_DIR, f"{model_name}_model.pth")
        print(f"📁 Model will be saved to: {MODEL_SAVE_PATH}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        best_val_acc = 0.0
        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            print(f"Epoch [{epoch}/{EPOCHS}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"💾 New best model saved at epoch {epoch}")

        # Load best model and test
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        test_metrics = test(model, test_loader, device)

        results_log.append(f"{model_name}\t{test_metrics['accuracy']:.4f}\t{test_metrics['precision']:.4f}\t{test_metrics['recall']:.4f}\t{test_metrics['f1']:.4f}\n")

    # Save results
    result_path = os.path.join(ROOT_DIR, "results_summary.txt")
    with open(result_path, "w") as f:
        f.writelines(results_log)

    print(f"\n✅ All done! Results saved to {result_path}")


if __name__ == "__main__":
    main()
