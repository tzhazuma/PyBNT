from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model

from common import normalize, denormalize, pixel_shuffle
from skimage.transform import resize

def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    x_in = Input(shape=(None, None, 3))
    #print(x_in.shape)
    x = Lambda(normalize)(x_in)

    x = b = Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b_in  = b
        b = Conv2D(num_filters, 3, padding='same', activation='relu')(b)
        b = Conv2D(num_filters, 3, padding='same')(b)
        if res_block_scaling:
            b = Lambda(lambda t: t * res_block_scaling)(b)
        b = Add()([b_in, b])

    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)

    x = Lambda(denormalize)(x)
    #x=resize(x, (x.shape[0],x.shape[1]/2, x.shape[2]/2,x.shape[3]), anti_aliasing=True)
    #print(x.shape)
    return Model(x_in, x, name="edsr")

# currently there is only a scale 4 model
def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')
    #x = upsample_1(x, 2, name='conv2d_1_scale_2')
    #x = upsample_1(x, 2, name='conv2d_2_scale_2')

    return x
def combi(num_filters=64,scale=2,num_res_blocks=6,res_block_scaling=None,combifilters=32):
    x1_in=Input(shape=(None,None,3));
    x2_in=Input(shape=(None,None,3));
    x1=Lambda(normalize)(x1_in)
    x2=Lambda(normalize)(x2_in)
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

    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)

    x1 = Lambda(denormalize)(x)
    x1=Conv2D(combifilters//2,3,padding="same")(x1)
    b1=x1;
    x2=Conv2D(combifilters//2,3,padding="same")(x2);
    b2=x2;
    x1=Conv2D(combifilters,3,padding="same")(x1);
    x2=Conv2D(combifilters,3,padding="same")(x2);
    x1=Concatenate()([x1,b1]);
    x2=Concatenate()([x2,b2]);
    x=Add()([x1,x2]);
    x=Conv2D(combifilters,3,padding="same")(x);
    x=Conv2D(combifilters//2,3,padding="same")(x);
    x=Conv2D(3,3,padding="same")(x);
    x=Lambda(denormalize)(x);
    return Model(inputs=[x1_in,x2_in],outputs=x,name="combi");