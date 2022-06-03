import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
import seaborn as sns
from PIL import Image

characters = ["ا","ب","ت","ث","ج","ح","خ","د","ذ","ر","ز","س","ش","ص","ض","ط","ظ","ع","غ","ف","ق","ك","ل","م","ن","ه","و","ي"]
characters_dict = dict(zip(np.arange(0,len(characters)), characters))

show_fig_names = [''] * len(characters)
fig_dir = './archive/Test Images 3360x32x32/test'

if __name__ == '__main__':
    fig_names = os.listdir(fig_dir)
    cnt = 0
    for i, name in enumerate(fig_names):
        name_without_subfix = os.path.splitext(name)[0]
        ss = name_without_subfix.split('_')
        label = int(ss[-1]) - 1
        if show_fig_names[label] == '':
            cnt = cnt + 1
            show_fig_names[label] = name 
        if name == len(characters):
            break
    print(show_fig_names)

    fig, ax = plt.subplots(4, 7, figsize=(12,8))
    i = 0
    for row in range(4):
        for col in range(7):
            plt.sca(ax[row, col])
            plt.title(f'label = {characters[i]}')
            assert os.path.exists(fig_dir + '/' + show_fig_names[i]), "figure does not exist in {}".format(fig_dir + '/' + show_fig_names[i])
            img = plt.imread(fig_dir + '/' + show_fig_names[i])
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            i += 1
    plt.savefig('viz.png', format='png', bbox_inches='tight', transparent=True, dpi=600)
    plt.show()



