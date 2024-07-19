import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import DeepVirFinderModel

class RandomDataset(Dataset):
    def __init__(self, num_samples=100, seq_length=150):
        self.data = torch.randn(num_samples, 4, seq_length)
        self.labels = torch.randint(0, 2, (num_samples, 1)).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train(model, device, train_loader, optimizer, epochs=10):
    criterion = nn.BCELoss()
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepVirFinderModel(seq_length=150).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = RandomDataset()
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    train(model, device, train_loader, optimizer)

    # 保存模型
    torch.save(model.state_dict(), "D:\\pythonProjectLibrary\\DeepVirFinderPytorch\\models\\model.pth")
