from typing import Tuple

from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, Dense


def build_model(input_shape: Tuple, feature_extractor="MobileNetV2") -> Model:
    """build dldl-v2 model

    Args:
        input_shape (Tuple): input image shape
        feature_extractor (str, optional): keras.applications model name. Defaults to "MobileNetV2".

    Returns:
        Model: dldl-v2 model
    """
    base_model = getattr(applications, feature_extractor)(
        include_top=False,
        input_shape=input_shape,
        weights="imagenet",
        pooling=None
    )
    features = base_model.output
    x = MaxPooling2D()(features)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=256, activation='relu', name="pred_dense")(x)
    x = Dense(units=100, activation="softmax", name="pred_label_dist")(x)
    v_x = Dense(units=1, activation="sigmoid", name="pred_age")(x)
    model = Model(inputs=base_model.input, outputs=[x, v_x])
    return model