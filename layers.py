import math

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from config import *

augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.02),  # factor
        layers.RandomZoom(height_factor=0.2, width_factor=0.2)
    ],
    name="data_augmentation"
)


def mlp(x, hidden, dropout):
    for units in hidden:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout)(x)
    return x


# patch tokenization layer
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        # print(f'patches: {patches.shape}')
        patch_dims = patches.shape[-1]  # patches are added depth wise
        patches = tf.reshape(patches, [batch, -1, patch_dims])
        return patches


# shifted patch tokenization layer
class ShiftedPatches(layers.Layer):
    def __init__(self, image_size=image_size, patch_size=patch_size, num_patches=num_patches,
                 projection_dim=projection_dim, regular=False):
        super(ShiftedPatches, self).__init__()
        self.regular = regular
        self.image_size = image_size
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.flatten_patches = layers.Reshape((num_patches, -1))
        self.projection = layers.Dense(units=projection_dim)
        self.layer_norm = layers.LayerNormalization(layer_norm)

    def crop_shift_pad(self, images, mode):
        # diagonally shifted images
        mode_map = {
            "left-up": [1, 1, 0, 0],
            "left-down": [0, 1, 1, 0],
            "right-up": [1, 0, 0, 1],
            "right-down": [0, 0, 1, 1]
        }
        flags = mode_map[mode]
        vals = []
        for flag in flags:
            if flag:
                vals.append(self.half_patch)
            else:
                vals.append(flag)
        crop_height, crop_width, shift_height, shift_width = vals

        # cropped images
        crop = tf.image.crop_to_bounding_box(
            images,
            offset_height=crop_height,
            offset_width=crop_width,
            target_height=self.image_size - self.half_patch,
            target_width=self.image_size - self.half_patch
        )
        shift_pad = tf.image.pad_to_bounding_box(
            crop,
            offset_height=shift_height,
            offset_width=shift_width,
            target_height=self.image_size,
            target_width=self.image_size
        )
        return shift_pad

    def call(self, images):
        if not self.regular:
            # shifted images with regular images
            images = tf.concat(
                [
                    images,
                    self.crop_shift_pad(images, mode="left-up"),
                    self.crop_shift_pad(images, mode="left-down"),
                    self.crop_shift_pad(images, mode="right-up"),
                    self.crop_shift_pad(images, mode="right-down")
                ],
                axis=-1
            )
            # flattened pacthes
            patches = tf.image.extract_patches(
                images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID"
            )
            patches = self.flatten_patches(patches)
            if not self.regular:
                # layer normalized flat patches
                patches = self.layer_norm(patches)
            '''
                tokens = self.projection(tokens)
            else:
                # linearly projected flat patches
                tokens = self.projection(flat_patches)
            '''
            return patches


# patch encoding layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(0, self.num_patches, 1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


# locality self attention - multi head
class MultiHeadLSA(layers.MultiHeadAttention):
    def __init__(self):
        super(MultiHeadLSA, self).__init__()
        # trainable temperature term, initial value equal to root of key dimension
        self.temp = tf.Variable(math.sqrt(float(self._key_dim)), trainable=True)

    def compute(self, query, key, value, attn_mask=None, training=None):
        query = tf.multiply(query, 1.0 / self.temp)
        attn_scores = tf.einsum(self._dot_product_equation, key, query)
        attn_scores = self._masked_softmax(attn_scores, attn_mask)
        attn_scores_dropout = self._dropout_layer(
            attn_scores, training=training
        )
        attn_output = tf.einsum(
            self._combine_equation, attn_scores_dropout, value
        )
        return attn_output, attn_scores


# diagonal attention mask
diag_attn_mask = 1 - tf.eye(num_patches)
diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)
