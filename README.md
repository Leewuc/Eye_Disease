# Eye Disease Classification

This project provides a simple convolutional neural network for classifying eye disease risk from retinal fundus images. The original implementation was a Jupyter notebook. The codebase is now split into modular scripts inside `src`.

## Directory Structure

- `src/dataset.py` – utilities for loading images and labels
- `src/model.py` – model architecture definition
- `src/train.py` – training script tying everything together
- `Disease.ipynb` – original notebook with the same workflow

## Quick Start

1. Install dependencies using `pip install -r requirements.txt`.
2. Run the training script with paths to your dataset:

```bash
python src/train.py \
  --train_dir <path-to-train-images> \
  --val_dir <path-to-val-images> \
  --test_dir <path-to-test-images> \
  --train_labels <path-to-train-csv> \
  --val_labels <path-to-val-csv> \
  --test_labels <path-to-test-csv>
```

The model will train, evaluate on the test set, show visualizations, and save the trained model as `model.h5`.

## Notes

Adjust the number of epochs or other hyperparameters in `src/train.py` as needed.
