import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from model import resnet34, resnet50, resnet101
from ArabicCharactersDataset import ArabicCharactersDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics import Accuracy


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

    test_accuracy_metric = Accuracy()
    acc = 0.0
    with torch.no_grad():
        test_bar = tqdm(test_dataloader)
        for test_data in test_bar:
            test_images, test_labels = test_data
            outputs = model(test_images.to(device).to(torch.float32))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_labels.to(device)).sum().item()
            batch_acc = test_accuracy_metric(predict_y, test_labels)
    test_accuracy = acc/test_num
    print(test_accuracy)
    print(test_accuracy_metric.compute())

    # for data in test_bar:
    #     sample, label = data
    #     # sample = torch.unsqueeze(sample, dim=0)  # expand batch dimension
    #     # labels.append(label)
    #     with torch.no_grad():
    #         outputs = model(sample.to(device).to(torch.float32))
    #         predict_y = torch.max(outputs, dim=1)[1]
    #         acc += torch.eq(predict_y, label.to(device)).sum().item()
    #         # predict = int(torch.max(predict, 1).indices[0])
    #         # predicts.append(predict)
    # test_accuarcy = acc/test_num
    # print(test_accuarcy)
    # labels = np.array(labels)
    # predicts = np.array(predicts)
    # pred_dir = './data/pred/'
    # if not os.path.exists(pred_dir):
    #     os.system('mkdir -p ./data/pred/')
    # label_path = os.path.join(pred_dir, 'label.npy')
    # pred_path = os.path.join(pred_dir, 'pred.npy')
    # np.save(label_path, labels)
    # np.save(pred_path, predicts)