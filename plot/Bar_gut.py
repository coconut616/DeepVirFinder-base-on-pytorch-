import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

def compute_auprc(file_path):
    y_true = []
    y_scores = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            y_true.append(int(parts[0]))
            y_scores.append(float(parts[1]))

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)

# Dictionary to hold AUPRC values
auprc_dict = {
    '150BP': compute_auprc('C:\\Users\\Ann\\Desktop\\DeepVirFinder\\test_homology\\gut\\gut_150BP_precision_recall_curve.txt'),
    '300BP': compute_auprc('C:\\Users\\Ann\\Desktop\\DeepVirFinder\\test_homology\\gut\\gut_300BP_precision_recall_curve.txt'),
    '500BP': compute_auprc('C:\\Users\\Ann\\Desktop\\DeepVirFinder\\test_homology\\gut\\gut_500BP_precision_recall_curve.txt'),
    '1000BP': compute_auprc('C:\\Users\\Ann\\Desktop\\DeepVirFinder\\test_homology\\gut\\gut_1000BP_precision_recall_curve.txt')
}

# Plotting the AUPRC as a bar chart
bp_sizes = list(auprc_dict.keys())
auprc_values = [auprc_dict[bp] for bp in bp_sizes]

plt.bar(bp_sizes, auprc_values, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Base Pair Size')
plt.ylabel('AUPRC')
plt.title('AUPRC by BP Size for DeepVirFinder')
plt.show()
