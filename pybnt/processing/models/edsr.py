"""EDSR and Combi model architectures for super-resolution."""
from tensorflow.python.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
)
from tensorflow.python.keras.models import Model

from pybnt.processing.models.common import normalize, denormalize, pixel_shuffle


def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    """Build EDSR (Enhanced Deep Super-Resolution) model.

    Parameters
    ----------
    scale : int
        Upscaling factor (2, 3, or 4).
    num_filters : int
        Number of convolution filters.
    num_res_blocks : int
        Number of residual blocks.
    res_block_scaling : float or None
        Residual block scaling factor.

    Returns
    -------
    tensorflow.keras.Model
    """
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    x = b = Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b_in = b
        b = Conv2D(num_filters, 3, padding='same', activation='relu')(b)
        b = Conv2D(num_filters, 3, padding='same')(b)
        if res_block_scaling:
            b = Lambda(lambda t: t * res_block_scaling)(b)
        b = Add()([b_in, b])

    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = _upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)

    x = Lambda(denormalize)(x)
    return Model(x_in, x, name="edsr")


def _upsample(x, scale, num_filters):
    """Upsample layer for EDSR, supporting scale 2, 3, or 4."""

    def _upsample_1(x_in, factor, **kwargs):
        x_out = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x_in)
        return Lambda(pixel_shuffle(scale=factor))(x_out)

    if scale == 2:
        x = _upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = _upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = _upsample_1(x, 2, name='conv2d_1_scale_2')
        x = _upsample_1(x, 2, name='conv2d_2_scale_2')

    return x


def combi(num_filters=64, scale=2, num_res_blocks=6,
          res_block_scaling=None, combifilters=32):
    """Build Combi model with dual input (image + reference).

    Parameters
    ----------
    num_filters : int
        Base convolution filters.
    scale : int
        Upscaling factor.
    num_res_blocks : int
        Residual blocks in the EDSR branch.
    res_block_scaling : float or None
        Residual block scaling.
    combifilters : int
        Filters for the combination layers.

    Returns
    -------
    tensorflow.keras.Model
    """
    x1_in = Input(shape=(None, None, 3))
    x2_in = Input(shape=(None, None, 3))
    x1 = Lambda(normalize)(x1_in)
    x2 = Lambda(normalize)(x2_in)

    x = b = Conv2D(num_filters, 3, padding='same')(x1)
    for i in range(num_res_blocks):
        b_in = b
        b = Conv2D(num_filters, 3, padding='same', activation='relu')(b)
        b = Conv2D(num_filters, 3, padding='same')(b)
        if res_block_scaling:
            b = Lambda(lambda t: t * res_block_scaling)(b)
        b = Add()([b_in, b])

    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = _upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)

    x1 = Lambda(denormalize)(x)
    x1 = Conv2D(combifilters // 2, 3, padding="same")(x1)
    b1 = x1
    x2 = Conv2D(combifilters // 2, 3, padding="same")(x2)
    b2 = x2
    x1 = Conv2D(combifilters, 3, padding="same")(x1)
    x2 = Conv2D(combifilters, 3, padding="same")(x2)
    x1 = Concatenate()([x1, b1])
    x2 = Concatenate()([x2, b2])
    x = Add()([x1, x2])
    x = Conv2D(combifilters, 3, padding="same")(x)
    x = Conv2D(combifilters // 2, 3, padding="same")(x)
    x = Conv2D(3, 3, padding="same")(x)
    x = Lambda(denormalize)(x)
    return Model(inputs=[x1_in, x2_in], outputs=x, name="combi")
