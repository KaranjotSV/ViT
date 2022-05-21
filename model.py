import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

from custom import augmentation, ShiftedPatches, PatchEncoder, mlp, MultiHeadLSA, diag_attn_mask
from config import *

classes = 100
in_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

print(f'training: {x_train.shape} - labels: {y_train.shape}')
print(f'testing: {x_test.shape} - labels: {y_test.shape}')


def create_ViT(SPT=True, LSA=True, MASKING=True, TRAIN_TAU=True):
    print(f'training - SPT: {SPT}, LSA: {LSA}, MASKING: {MASKING}, TRAIN_TAU: {TRAIN_TAU}')
    inputs = layers.Input(in_shape)
    augmented = augmentation(inputs)  # augment data
    patches = ShiftedPatches(SPT=SPT)(augmented)  # create patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)  # encode patches

    # create multiple layers of transformer block
    for _ in range(transformer_layers):
        # normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # create a multi-head attention layer
        if LSA and (MASKING or TRAIN_TAU):
            if MASKING:
                attn_output = MultiHeadLSA(
                    num_heads=num_heads, key_dim=projection_dim, TRAIN_TAU=TRAIN_TAU, dropout=0.1
                )(x1, x1, attention_mask=diag_attn_mask)
            elif not MASKING:
                attn_output = MultiHeadLSA(
                    num_heads=num_heads, key_dim=projection_dim, TRAIN_TAU=TRAIN_TAU, dropout=0.1
                )(x1, x1)
        else:
            attn_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
        # skip connection 1
        x2 = layers.Add()([attn_output, encoded_patches])
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


def run(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer,
        keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy")
        ]
    )

    checkpoint_path = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback]
    )

    model.load_weights(checkpoint_path)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)

    print(f'test accuracy: {accuracy * 100:.2f}')
    print(f'test top 5 accuracy: {top_5_accuracy * 100:.2f}')

    return history
