import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from model import resnet34, resnet50, resnet101
from ArabicCharactersDataset import ArabicCharactersDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

weights_path = './arabic.pth'
test_dir = './archive/Test Images 3360x32x32/test'

batch_size = 32

if __name__ == '__main__':
    assert os.path.exists(weights_path), "Weights does not exist in {}".format(weights_path)
    assert os.path.exists(test_dir), "Test data directory does not exist in {}".format(test_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print("Using {} dataloader workers every process".format(num_workers))

    test_dataset = ArabicCharactersDataset(test_dir)
    test_num = len(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, 
                                 shuffle=False, num_workers=num_workers)

    example_img, example_label = test_dataset[0]
    c, h, w = example_img.shape

    model = resnet34(num_classes=28)
    model.to(device=device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    test_accuracy_metric = Accuracy(average='macro', num_classes=28).to(device)
    test_precision_metric = Precision(average='macro', num_classes=28).to(device)
    test_recall_metric = Recall(average='macro', num_classes=28).to(device)
    test_f1score_metric = F1Score(average='macro', num_classes=28).to(device)
    test_confusion_matrix = ConfusionMatrix(num_classes=28).to(device)
    acc = 0.0
    with torch.no_grad():
        test_bar = tqdm(test_dataloader)
        for test_data in test_bar:
            test_images, test_labels = test_data
            outputs = model(test_images.to(device).to(torch.float32))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_labels.to(device)).sum().item()
            batch_acc = test_accuracy_metric(outputs.to(device), test_labels.to(device))
            batch_prc = test_precision_metric(outputs.to(device), test_labels.to(device))
            batch_rcl = test_recall_metric(outputs.to(device), test_labels.to(device))
            batch_f1 = test_f1score_metric(outputs.to(device), test_labels.to(device))
            batch_cm = test_confusion_matrix(outputs.to(device), test_labels.to(device))
    test_accuracy = acc/test_num
    print(test_accuracy)
    print("Accuracy:", test_accuracy_metric.compute())
    print("Precision:", test_precision_metric.compute())
    print("Recall:", test_recall_metric.compute())
    print("F1 Score:", test_f1score_metric.compute())
    print("Confusion matrix: \n", test_confusion_matrix.compute())

    # Set photo parameters
    # mpl.rcParams['font.sans-serif'] = 'Times New Roman'
    # mpl.rcParams['axes.unicode_minus'] = False

    f, ax = plt.subplots()
    cm = test_confusion_matrix.compute()
    cm = cm.cpu().numpy()
    # cm = np.array(test_confusion_matrix.compute())
    cm = np.around(cm/sum(cm), 28)

    cmap = sns.color_palette("Spectral", as_cmap=True)
    sns.heatmap(cm, annot=False, ax=ax, fmt='.4f', cmap=cmap)
    # ticklabels = list(range(1,28+1))
    # ticklabels = [str(i) for i in ticklabels]
    # ax.set(xticklabels=ticklabels, yticklabels=ticklabels)
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=14, fontweight='bold')
    plt.savefig('Confusion_Matrix.png', format='png', bbox_inches='tight', transparent=True, dpi=600)
    # plt.show()