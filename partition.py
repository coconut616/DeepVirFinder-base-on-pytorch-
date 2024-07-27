length = 150
input_file = "D:\\pythonProjectLibrary\\DeepVirFinderPytorch\\test_homology\\gut_output_match2.fasta"
output_file = "D:\\pythonProjectLibrary\\DeepVirFinderPytorch\\test_homology\\gut_output_match2_150.fasta"

def partition_string(s, length):
    if len(s) < length:
        return s  # return the whole sequence if it's shorter than the specified length
    return s[:length]

count = 0
with open(input_file, 'r') as f:
    with open(output_file, 'w') as f2:
        b = False
        label = ''
        seq = ''
        for line in f:
            if line.startswith(">"):
                if b:
                    count += 1
                    f2.write(label)
                    f2.write(partition_string(seq, length))
                    f2.write("\n")
                    print(f"done{count}")
                    seq = ''
                b = True
                label = line
            else:
                seq += line.strip()
        if b:
            count += 1
            f2.write(label)
            f2.write(partition_string(seq, length))
            f2.write("\n")
            print(f"done{count}")

print(count)
