import multiprocessing

import albumentations as A
import tensorflow as tf
from dldl_v2.generator import ImageGenerator
from dldl_v2.model import build_model
from tensorflow.keras.optimizers import Adam

if __name__ == "__main__":

    INPUT_SHAPE = (224, 224, 3)
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    TRAIN_TRANSFORMER = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=1.0),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, p=1.0),
            A.CenterCrop(p=1.0, height=224, width=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.Resize(height=INPUT_SHAPE[1], width=INPUT_SHAPE[0]),
        ]
    )

    train_generator = ImageGenerator(BATCH_SIZE, "sample_train.csv", TRAIN_TRANSFORMER)

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

    # model.fit(train_generator, epochs=10, workers=multiprocessing.cpu_count())
