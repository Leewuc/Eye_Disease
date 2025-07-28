import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from dataset import load_images_and_labels
from model import build_model


def display_samples(images, true_labels, pred_labels, sample_count=10):
    plt.figure(figsize=(20, 20))
    for i in range(sample_count):
        plt.subplot(5, 2, i + 1)
        plt.imshow(images[i])
        plt.title(f"True: {true_labels[i]}, Pred: {pred_labels[i][0]}")
        plt.axis('off')
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Train eye disease classifier")
    parser.add_argument("--train_dir", required=True, help="Directory of training images")
    parser.add_argument("--val_dir", required=True, help="Directory of validation images")
    parser.add_argument("--test_dir", required=True, help="Directory of test images")
    parser.add_argument("--train_labels", required=True, help="CSV file with training labels")
    parser.add_argument("--val_labels", required=True, help="CSV file with validation labels")
    parser.add_argument("--test_labels", required=True, help="CSV file with test labels")
    return parser.parse_args()


def main():
    args = parse_args()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    train_labels = pd.read_csv(args.train_labels)
    val_labels = pd.read_csv(args.val_labels)
    test_labels = pd.read_csv(args.test_labels)

    X_train, y_train = load_images_and_labels(args.train_dir, train_labels)
    X_val, y_val = load_images_and_labels(args.val_dir, val_labels)
    X_test, y_test = load_images_and_labels(args.test_dir, test_labels)

    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    model = build_model()

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=20
    )

    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    px.imshow(cm, text_auto=True, title="Confusion Matrix").show()

    fig = go.Figure()
    epochs = range(1, len(history.history['accuracy']) + 1)
    fig.add_trace(go.Scatter(x=list(epochs), y=history.history['accuracy'], mode='lines', name='Train Accuracy'))
    fig.add_trace(go.Scatter(x=list(epochs), y=history.history['val_accuracy'], mode='lines', name='Validation Accuracy'))
    fig.update_layout(title='Model Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy')
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(epochs), y=history.history['loss'], mode='lines', name='Train Loss'))
    fig.add_trace(go.Scatter(x=list(epochs), y=history.history['val_loss'], mode='lines', name='Validation Loss'))
    fig.update_layout(title='Model Loss', xaxis_title='Epoch', yaxis_title='Loss')
    fig.show()

    display_samples(X_test, y_test, y_pred, sample_count=10)

    model.save('model.h5')


if __name__ == '__main__':
    main()
