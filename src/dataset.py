import os
import pandas as pd
import numpy as np
import cv2


def load_images_and_labels(image_dir: str, labels_df: pd.DataFrame):
    """Load images and their labels from a directory."""
    images = []
    labels = []
    for _, row in labels_df.iterrows():
        img_path = os.path.join(image_dir, f"{row['ID']}.png")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(row['Disease_Risk'])
    return np.array(images), np.array(labels)
