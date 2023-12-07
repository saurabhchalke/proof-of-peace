import torch
import torch.optim as optim
import torch.nn as nn
from model import NeutronDetectorCNN
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np


def load_data():
    data, labels = np.load("data.npy"), np.load("labels.npy")
    X = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.float32)
    return TensorDataset(X, y)


def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.to(device)  # Move model to the selected device
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data, target = data.to(device), target.to(
                device
            )  # Move data and target to the selected device
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dataset = load_data()
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.2, random_state=42
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = NeutronDetectorCNN(image_size=10)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, device)

    # Saving the trained model
    torch.save(model.state_dict(), "neutron_detector.pth")
