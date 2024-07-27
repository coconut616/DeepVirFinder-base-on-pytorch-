import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
import os
import model  # 确认导入正确的模型模块

class EncodedDataset(Dataset):
    def __init__(self, tensor_dir):
        self.tensor_dir = tensor_dir
        self.file_names = [f for f in os.listdir(tensor_dir) if f.endswith('_fwd_tensor.pt')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        seq_id = self.file_names[idx].replace('_fwd_tensor.pt', '')
        fwd_tensor = torch.load(os.path.join(self.tensor_dir, f"{seq_id}_fwd_tensor.pt"))
        rev_tensor = torch.load(os.path.join(self.tensor_dir, f"{seq_id}_rev_tensor.pt"))
        label_tensor = torch.load(os.path.join(self.tensor_dir, f"{seq_id}_label_tensor.pt"))
        return fwd_tensor, rev_tensor, label_tensor

def collate_fn(batch):
    fwd_tensors, rev_tensors, labels = zip(*batch)

    max_len = max(tensor.shape[2] for tensor in fwd_tensors)
    padded_fwd_tensors = [torch.nn.functional.pad(tensor, (0, max_len - tensor.shape[2])) for tensor in fwd_tensors]
    padded_rev_tensors = [torch.nn.functional.pad(tensor, (0, max_len - tensor.shape[2])) for tensor in rev_tensors]

    fwd_batch = torch.cat(padded_fwd_tensors, dim=0)
    rev_batch = torch.cat(padded_rev_tensors, dim=0)
    label_batch = torch.stack(labels)

    return fwd_batch, rev_batch, label_batch

def main():
    # 实例化模型和数据集
    model_instance = model.DeepVirFinderModel()

    # 假设已经处理并保存了编码后的张量数据
    tensor_dir = 'D:\\pythonProjectLibrary\\DeepVirFinderPytorch\\data\\tensors'  # 修改为实际路径
    dataset = EncodedDataset(tensor_dir)

    # 划分数据集
    total_size = len(dataset)
    train_size = int(0.85 * total_size)
    valid_size = int(0.05 * total_size)
    test_size = total_size - train_size - valid_size

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    # 创建 DataLoader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # 保存初始模型
    os.makedirs('D:\\pythonProjectLibrary\\DeepVirFinderPytorch\\models', exist_ok=True)
    torch.save(model_instance.state_dict(), 'D:\\pythonProjectLibrary\\DeepVirFinderPytorch\\models\\initial_model.pth')

    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_instance.parameters(), lr=0.001)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_instance.to(device)

    # 训练模型
    num_epochs = 10  # 初步选择一个较小的值
    best_valid_loss = float('inf')

    for epoch in range(num_epochs):
        model_instance.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            fwd_inputs, rev_inputs, labels = data
            fwd_inputs, rev_inputs, labels = fwd_inputs.to(device), rev_inputs.to(device), labels.to(device)

            # 调整输入形状
            fwd_inputs = fwd_inputs.squeeze(1)
            rev_inputs = rev_inputs.squeeze(1)

            optimizer.zero_grad()

            outputs = model_instance.forward_backward_avg(fwd_inputs, rev_inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        # 在每个epoch结束时进行验证
        model_instance.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for data in valid_loader:
                fwd_inputs, rev_inputs, labels = data
                fwd_inputs, rev_inputs, labels = fwd_inputs.to(device), rev_inputs.to(device), labels.to(device)

                # 调整输入形状
                fwd_inputs = fwd_inputs.squeeze(1)
                rev_inputs = rev_inputs.squeeze(1)

                outputs = model_instance.forward_backward_avg(fwd_inputs, rev_inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {valid_loss:.4f}')

        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model_instance.state_dict(), 'D:\\pythonProjectLibrary\\DeepVirFinderPytorch\\models\\best_model.pth')

    print('Finished Training')

if __name__ == '__main__':
    main()
