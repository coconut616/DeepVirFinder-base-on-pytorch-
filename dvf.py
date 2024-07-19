import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys
from Bio import SeqIO
import argparse

from model import DeepVirFinderModel
from encode import encode_seq


class SequenceDataset(Dataset):
    def __init__(self, sequences, reverse_sequences):
        assert len(sequences) == len(reverse_sequences)
        self.sequences = sequences
        self.reverse_sequences = reverse_sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.reverse_sequences[idx]


def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepVirFinderModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded successfully from", model_path)
    return model, device


def predict(model, device, dataset):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    results = []
    with torch.no_grad():
        for fwd_data, rev_data in loader:
            fwd_data, rev_data = fwd_data.to(device), rev_data.to(device)
            output = model.forward_backward_avg(fwd_data.squeeze(0), rev_data.squeeze(0))
            results.append(output.cpu().numpy())
    return results


def main(model_path, fasta_path, output_path):
    print("Starting prediction process...")
    model, device = load_model(model_path)
    sequences = []
    print("Loading sequences from", fasta_path)

    reverse_sequences = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        encoded_fwd, encoded_rev = encode_seq(str(record.seq))
        tensor_fwd = torch.tensor(encoded_fwd, dtype=torch.float32).unsqueeze(0)
        tensor_rev = torch.tensor(encoded_rev, dtype=torch.float32).unsqueeze(0)
        sequences.append(tensor_fwd)
        reverse_sequences.append(tensor_rev)

    dataset = SequenceDataset(sequences, reverse_sequences)
    results = predict(model, device, dataset)

    with open(output_path, 'w') as f:
        print("Prediction complete. Results:")
        for i, score in enumerate(results):
            label = "Virus" if score > 0.5 else "Bacteria"
            f.write(f"Sequence {i + 1}: Score = {score[0][0]}, Label = {label}\n")
            print(f"Sequence {i + 1}: Score = {score[0][0]}, Label = {label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepVirFinder: Identifying viral sequences using deep learning.')
    parser.add_argument('-i', '--input', required=True, help='Input FASTA file path')
    parser.add_argument('-m', '--model', required=True, help='Path to the trained model file')
    parser.add_argument('-o', '--output', required=True, help='Path to the output result file')
    args = parser.parse_args()

    main(args.model, args.input, args.output)
