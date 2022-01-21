import tensorflow as tf
from dldl_v2.model import build_model
from tensorflow.keras.optimizers import Adam

if __name__ == "__main__":

    INPUT_SHAPE = (224, 224, 3)
    LEARNING_RATE = 1e-3

    model = build_model(INPUT_SHAPE)
    optimizer = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0, epsilon=1e-8)
    model.compile(
        optimizer=optimizer,
        loss=[
            tf.keras.losses.kullback_leibler_divergence,
            tf.keras.losses.mean_absolute_error,
        ],
    )
    print(model.summary())
