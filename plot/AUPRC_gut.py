import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

def plot_precision_recall_curve(file_path, label):
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
    auprc = auc(recall, precision)

    plt.plot(recall, precision, marker='.', label=f'{label} (AUPRC = {auprc:.2f})')

# 读取并绘制各个BP的曲线
plot_precision_recall_curve('C:\\Users\\Ann\\Desktop\\DeepVirFinder\\test_homology\\gut\\gut_150BP_precision_recall_curve.txt', 'Precision-Recall curve (150BP)')
plot_precision_recall_curve('C:\\Users\\Ann\\Desktop\\DeepVirFinder\\test_homology\\gut\\gut_300BP_precision_recall_curve.txt', 'Precision-Recall curve (300BP)')
plot_precision_recall_curve('C:\\Users\\Ann\\Desktop\\DeepVirFinder\\test_homology\\gut\\gut_500BP_precision_recall_curve.txt', 'Precision-Recall curve (500BP)')
plot_precision_recall_curve('C:\\Users\\Ann\\Desktop\\DeepVirFinder\\test_homology\\gut\\gut_1000BP_precision_recall_curve.txt', 'Precision-Recall curve (1000BP)')
plot_precision_recall_curve('C:\\Users\\Ann\\Desktop\\DeepVirFinder\\test_homology\\gut\\gut_3000BP_precision_recall_curve.txt', 'Precision-Recall curve (3000BP)')


# 绘制曲线
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for DeepVirfinder')
plt.legend()
plt.show()
