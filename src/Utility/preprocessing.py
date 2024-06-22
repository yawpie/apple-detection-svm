import cv2


def perform_grayscale(images):
    grayscaled = {label: [] for label in images.keys()}

    for label in images.keys():
        for image in images[label]:
            grayscaled[label].append(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    return grayscaled
