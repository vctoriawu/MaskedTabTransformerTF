import numpy as np
import tensorflow as tf
from tensorflow.keras.activations import selu
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
import matplotlib.pyplot as plt
import pandas as pd

def build_mlp(input_dim, factors, dropout):
    hidden_units = [input_dim // f for f in factors]

    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(BatchNormalization()),
        mlp_layers.append(Dense(units, activation=selu))
        mlp_layers.append(Dropout(dropout))

    return tf.keras.Sequential(mlp_layers)


def generate_mask(x, p_replace=0.2):
    m = np.random.choice([False, True], size=x.shape, p=[1-p_replace, p_replace])
    return m

def generate_batch_mask(batch_size, num_features, p_replace):
    # Generate a single mask for each feature (column)
    mask = np.random.choice([False, True], size=num_features, p=[1-p_replace, p_replace])

    # Repeat the mask for each row in the batch
    batch_mask = np.tile(mask, (batch_size, 1))

    return batch_mask

def corrupt_dataset(x, p_replace=0.2):
    mask = generate_mask(x, p_replace=p_replace)
    shuffled_data = np.random.permutation(x)
    corrupted_data = x.copy()
    corrupted_data = corrupted_data.mask(mask, shuffled_data)
    new_mask = (corrupted_data == x).values
    return corrupted_data, new_mask

def corrupt_dataset_batchwise(x, p_replace=0.2, replacement_value=None, shuffle=False, batch_size=512):
    num_samples, num_features = x.shape

    if shuffle:
        # Randomly shuffle the original data
        shuffled_indices = np.random.permutation(num_samples)
        shuffled_data = x.iloc[shuffled_indices, :]
    else:
        shuffled_data = x

    # Iterate through batches and apply masks
    corrupted_data_list = []
    original_data_list = []
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_x = shuffled_data.iloc[start_idx:end_idx, :]

        # Generate a new mask for each batch
        mask = generate_batch_mask(batch_x.shape[0], num_features, p_replace)

        if replacement_value is None:
            corrupted_batch = np.where(mask[:, np.newaxis], batch_x, x.iloc[start_idx:end_idx, :])
        else:
            # Replace with a specified value
            corrupted_batch = np.where(mask, replacement_value, batch_x)

        corrupted_data_list.append(pd.DataFrame(corrupted_batch, columns=x.columns))
        original_data_list.append(x.iloc[start_idx:end_idx, :])

    corrupted_data = pd.concat(corrupted_data_list)
    original_data = pd.concat(original_data_list)
    return corrupted_data, original_data


def get_model_importances(importances, title="Importances"):
    imps_sorted = importances.mean().sort_values(ascending=False)
    
    plt.figure(figsize=(15,7))
    ax = imps_sorted.plot.bar()
    for p in ax.patches:
        ax.annotate(str(np.round(p.get_height(), 4)), (p.get_x(), p.get_height() * 1.01))
    plt.title(title)
    plt.show()
    
    return imps_sorted
