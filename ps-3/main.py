import os
from pathlib import Path
import numpy as np
import cv2

from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pylab as plt

def vec_to_img(vec, img_path):
    image_np = np.array(vec, dtype=np.uint8).reshape(28,28,1)
    cv2.imwrite(img_path, image_np)

def normalization(dataset):
    mean = np.mean(dataset, axis=0)

    return dataset - mean


if __name__ == "__main__":
    # Loading the MNIST dataset.
    # if not Path("images").exists():
    #     Path("images").mkdir(parents=True, exist_ok=True)
    #     num_file = 60000
    #     with open('train-images-idx3-ubyte', 'rb') as f:
    #         image_file = f.read()
    #     image_file = image_file[16:]

    #     for i in range(num_file):
    #         image_list = [int(item) for item in image_file[i*784:(i+1)*784]]
    #         image_np = np.array(image_list, dtype=np.uint8).reshape(28,28,1)
    #         save_name = os.path.join('images', '{}_{}.jpg'.format('train', i))
    #         cv2.imwrite(save_name, image_np)
    num_ins = 60000

    vector_set = []
    with open('train-images-idx3-ubyte', 'rb') as f:
        file = f.read()
    
    file = file[16:]
    for i in range(num_ins):
        vec =  [int(item) for item in file[i*784:(i+1)*784]]
        vector_set.append(vec)

    vector_set = np.array(vector_set)
    print(vector_set.shape)

    normal_vector_set = normalization(vector_set)
    # normal_vector_set = vector_set
    
    # pro-4-1 & pro-4-3 build PCA models preserve 8, 40, 156, 392, 627, 745, 776 information.
    # for n_component in [8, 40, 156, 392, 627, 745, 776]:
    #     pca_l = PCA(n_components=n_component)
    #     normal_vector_set_pca = pca_l.fit_transform(normal_vector_set)
    #     normal_vector_set_pca = pca_l.inverse_transform(normal_vector_set_pca)
    #     for i in range(3):
    #         vec_to_img(normal_vector_set_pca[i], f'outputs/{n_component}-{i}_pca.png')

    pca_l = PCA(n_components=100)
    pca_l.fit(normal_vector_set)
    plt.figure(figsize=(50,50))
    for i in range(10):
        for j in range(10):
            count = i * 10 + j
            plt.subplot(10,10,count+1) #要生成两行两列，这是第一个图plt.subplot('行','列','编号')
            plt.imshow(pca_l.components_[count].reshape(28,28,1))
    plt.savefig("result.png")
    print(pca_l.explained_variance_)