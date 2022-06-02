import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from model import resnet34, resnet50, resnet101
from ArabicCharactersDataset import ArabicCharactersDataset
import matplotlib.pyplot as plt


weights_path = './arabic.pth'
test_dir = './archive/Test Images 3360x32x32/test'

if __name__ == '__main__':
    # assert os.path.exists(weights_path), "Weights does not exist in {}".format(weights_path)
    assert os.path.exists(test_dir), "Test data directory does not exist in {}".format(test_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))

    test_dataset = ArabicCharactersDataset(test_dir)

    example_img, example_label = test_dataset[0]
    c, h, w = example_img.shape

    model = resnet50(num_classes=28)
    model.to(device=device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    labels = []
    predicts = []
    for i, data in enumerate(test_dataset):
        sample, label = data
        sample = torch.unsqueeze(sample, dim=0)  # expand batch dimension
        labels.append(label)
        with torch.no_grad():
            predict = model(sample.to(device))
            predict = int(torch.max(predict, 1).indices[0])
            predicts.append(predict)
    labels = np.array(labels)
    predicts = np.array(predicts)

    print(accuracy_score(labels, predicts))

    # pred_dir = './data/pred/'
    # if not os.path.exists(pred_dir):
    #     os.system('mkdir -p ./data/pred/')
    
    # label_path = os.path.join(pred_dir, 'label.npy')
    # pred_path = os.path.join(pred_dir, 'pred.npy')
    # np.save(label_path, labels)
    # np.save(pred_path, predicts)