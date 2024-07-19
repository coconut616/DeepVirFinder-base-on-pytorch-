from Bio import SeqIO
import numpy as np
import os
import torch


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



def process_file(file_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(file_name, 'r') as file:
        for record in SeqIO.parse(file, "fasta"):
            fwd_encoded, rev_encoded = encode_seq(str(record.seq))

            fwd_tensor = torch.tensor(fwd_encoded, dtype=torch.float32).unsqueeze(0)
            rev_tensor = torch.tensor(rev_encoded, dtype=torch.float32).unsqueeze(0)

            torch.save(fwd_tensor, os.path.join(output_dir, f"{record.id}_fwd_tensor.pt"))
            torch.save(rev_tensor, os.path.join(output_dir, f"{record.id}_rev_tensor.pt"))

if __name__ == "__main__":
    import sys
    file_name = sys.argv[1]
    output_dir = sys.argv[2]
    process_file(file_name, output_dir)
