from typing import Tuple

from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    ReLU,
)
from tensorflow.keras.models import Model


def build_model(input_shape: Tuple, compression_rate: float = 0.5) -> Model:
    """build dldl-v2 model

    Args:
        input_shape (Tuple): input image shape
        compression_rate (float, optional): compression rate of vgg16 filter. Defaults to 0.5.

    Returns:
        Model: dldl-v2 model
    """

    def ThinAgeNet(input_shape: Tuple, compression_rate: float = 0.5):
        def conv_block(x, filter_num: int, name: str):
            x = Conv2D(
                filters=filter_num,
                kernel_size=(3, 3),
                padding="same",
                name=name + "_conv",
            )(x)
            x = BatchNormalization(name=name + "_bn")(x)
            x = ReLU(name=name + "_relu")(x)
            return x

        inputs = Input(shape=input_shape)

        # block1
        x = conv_block(inputs, filter_num=int(64 * compression_rate), name="block1_1")
        x = conv_block(x, filter_num=int(64 * compression_rate), name="block1_2")
        x = MaxPooling2D(name="block1_pool")(x)

        # block2
        x = conv_block(x, filter_num=int(128 * compression_rate), name="block2_1")
        x = conv_block(x, filter_num=int(128 * compression_rate), name="block2_2")
        x = MaxPooling2D(name="block2_pool")(x)

        # block3
        x = conv_block(x, filter_num=int(256 * compression_rate), name="block3_1")
        x = conv_block(x, filter_num=int(256 * compression_rate), name="block3_2")
        x = conv_block(x, filter_num=int(256 * compression_rate), name="block3_3")
        x = MaxPooling2D(name="block3_pool")(x)

        # block4
        x = conv_block(x, filter_num=int(512 * compression_rate), name="block4_1")
        x = conv_block(x, filter_num=int(512 * compression_rate), name="block4_2")
        x = conv_block(x, filter_num=int(512 * compression_rate), name="block4_3")
        x = MaxPooling2D(name="block4_pool")(x)

        # block5
        x = conv_block(x, filter_num=int(512 * compression_rate), name="block5_1")
        x = conv_block(x, filter_num=int(512 * compression_rate), name="block5_2")
        x = conv_block(x, filter_num=int(512 * compression_rate), name="block5_3")
        x = MaxPooling2D(name="block5_pool")(x)
        return Model(inputs=inputs, outputs=x)

    thin_age_net = ThinAgeNet(input_shape, compression_rate=compression_rate)
    x = GlobalAveragePooling2D()(thin_age_net.output)
    x = Dense(units=256, activation="relu", name="pred_dense")(x)
    x = Dense(units=100, activation="softmax", name="pred_label_dist")(x)
    v_x = Dense(units=1, activation="sigmoid", name="pred_age")(x)
    model = Model(inputs=thin_age_net.input, outputs=[x, v_x])
    return model
