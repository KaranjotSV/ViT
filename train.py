import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

from layers import augmentation, Patches, PatchEncoder, mlp
from config import *


# data
classes = 100
in_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
print(f'training: {x_train.shape} - labels: {y_train.shape}')
print(f'testing: {x_test.shape} - labels: {y_test.shape}')

augmentation.layers[0].adapt(x_train)

def create_ViT():
    inputs = layers.Input(in_shape)
    augmented = augmentation(inputs)  # augment data
    patches = Patches(patch_size)(augmented)  # create patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)  # encode patches

    # create multiple layers of transformer block
    for _ in range(transformer_layers):
        # normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # create a multi-head attention layer
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        # normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = mlp(x3, hidden=transformer_units, dropout=0.1)
        # skip connection 2
        encoded_patches = layers.Add()([x3, x2])

    # create a [batch_size, projection_dim] tensor
    rep = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    rep = layers.Flatten()(rep)
    rep = layers.Dropout(0.5)(rep)

    # MLP
    features = mlp(rep, hidden=mlp_head, dropout=0.5)
    # classify
    logits = layers.Dense(classes)(features)

    model = keras.Model(inputs=inputs, outputs=logits)
    return model

