import torch
import torch.nn as nn
from model import DeepVirFinderModel

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepVirFinderModel(seq_length=150).to(device)

    model_save_path = "D:\\pythonProjectLibrary\\DeepVirFinderPytorch\\models\\model.pth"
    torch.save(model.state_dict(), model_save_path)

    print(f"Model has been saved to {model_save_path}")
