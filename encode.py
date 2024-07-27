from Bio import SeqIO
import numpy as np
import os
import torch
import sys


def encode_seq(seq):
    seq_code = []
    rev_code = []

    for letter in seq.upper():
        if letter == 'A':
            fwd_code = [1, 0, 0, 0]
            rev_code.append([0, 0, 0, 1])  # T
        elif letter == 'C':
            fwd_code = [0, 1, 0, 0]
            rev_code.append([0, 0, 1, 0])  # G
        elif letter == 'G':
            fwd_code = [0, 0, 1, 0]
            rev_code.append([0, 1, 0, 0])  # C
        elif letter == 'T':
            fwd_code = [0, 0, 0, 1]
            rev_code.append([1, 0, 0, 0])  # A
        else:
            fwd_code = [0.25, 0.25, 0.25, 0.25]
            rev_code.append([0.25, 0.25, 0.25, 0.25])
        seq_code.append(fwd_code)

    rev_code.reverse()
    return np.array(seq_code).T, np.array(rev_code).T


def extract_label(description):
    if "Viruses" in description:
        return 0.0
    else:
        return 1.0  # or raise an error


def process_file(file_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(file_name, 'r') as file:
        records = list(SeqIO.parse(file, "fasta"))
        print(f"Found {len(records)} sequences in the input file.")
        sys.stdout.flush()

        for record in records:
            seq_id = record.id
            label = extract_label(record.description)
            if label == -1.0:
                print(f"Skipping sequence {seq_id} due to invalid label.")
                sys.stdout.flush()
                continue  # Skip sequences without valid labels

            fwd_encoded, rev_encoded = encode_seq(str(record.seq))

            fwd_tensor = torch.tensor(fwd_encoded, dtype=torch.float32).unsqueeze(0)
            rev_tensor = torch.tensor(rev_encoded, dtype=torch.float32).unsqueeze(0)
            label_tensor = torch.tensor(float(label), dtype=torch.float32).unsqueeze(0)

            torch.save(fwd_tensor, os.path.join(output_dir, f"{seq_id}_fwd_tensor.pt"))
            torch.save(rev_tensor, os.path.join(output_dir, f"{seq_id}_rev_tensor.pt"))
            torch.save(label_tensor, os.path.join(output_dir, f"{seq_id}_label_tensor.pt"))

            print(f"Processed sequence {seq_id} and saved tensors.")
            sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python encode3.py <input_file> <output_directory>")
        sys.stdout.flush()
    else:
        file_name = sys.argv[1]
        output_dir = sys.argv[2]
        print(f"Processing file {file_name} and saving to {output_dir}")
        sys.stdout.flush()
        process_file(file_name, output_dir)
        print("Processing completed.")
        sys.stdout.flush()
