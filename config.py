learning_rate = 0.001
weight_decay = 0.0001
batch = 256
epochs = 20
image_size = 72
patch_size = 6
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4  # multi-head attention

transformer_units = [projection_dim * 2,  projection_dim]  # size of transformer layers
transformer_layers = 8
mlp_head = [2048, 1024]  # final classifier
