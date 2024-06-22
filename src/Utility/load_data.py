import os
import cv2
from matplotlib import pyplot as plt
import numpy as np


def load_processed_data(folder_path) -> dict:
    images = {label: [] for label in os.listdir(folder_path)}
    for folders in os.listdir(folder_path):
        for files in os.listdir(os.path.join(folder_path, folders)):
            image = cv2.imread(os.path.join(folder_path, folders, files))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images[folders].append(image)

    return images


def resize_raw_data(folder_path: str, output_path: str):

    images = {label: [] for label in os.listdir(folder_path)}

    for folders in os.listdir(folder_path):
        for files in os.listdir(os.path.join(folder_path, folders)):
            filename = os.path.basename(files)
            image = cv2.imread(os.path.join(
                folder_path, folders, files), cv2.IMREAD_COLOR)
            # image = crop(image)
            image = cv2.resize(image, (150, 150))
            images[folders].append(image)
            write_path = output_path or os.path.join(folder_path, 'resized')
            write_images_and_labels(image, filename,  write_path, folders)


def write_images_and_labels(image, filename: str, folder_path: str, folder_title: str = None):
    path = folder_path
    if folder_title is not None:
        path = os.path.join(folder_path, folder_title)
        os.makedirs(path, exist_ok=True)
    cv2.imwrite(os.path.join(path, filename), image)


def write_images_from_dict(images: dict, path):
    for label in images.keys():
        for i, image in enumerate(images[label]):
            write_path = os.path.join(path, label)
            if not os.path.exists(write_path):
                os.makedirs(write_path, exist_ok=True)
            cv2.imwrite(os.path.join(write_path, f'{i}.png'), image)


def show_random_images(images: dict, label: str, amount: int = 5):
    plt.figure(figsize=(15, 5))
    sample = np.random.randint(0, len(images[label]), amount)
    for i, index in enumerate(sample):
        plt.suptitle(f'Label: {label}')
        plt.subplot(1, amount, i + 1)
        plt.imshow(images[label][index])
        plt.axis('off')
    plt.show()
