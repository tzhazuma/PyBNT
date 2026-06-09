"""Image dataset pipeline for super-resolution training."""
import os

import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE


def dataset(
    trainpathx,
    tranpathy,
    testpathx=None,
    testpathy=None,
    refpath=None,
    batch_size=16,
    repeat_count=None,
    random_transform=True,
    subset='train',
):
    """Create a tf.data.Dataset for super-resolution training.

    Parameters
    ----------
    trainpathx : str
        Path to low-resolution training images.
    tranpathy : str
        Path to high-resolution training images.
    testpathx : str or None
        Path to low-resolution validation images.
    testpathy : str or None
        Path to high-resolution validation images.
    refpath : str or None
        Path to reference (flair) images.
    batch_size : int
        Batch size.
    repeat_count : int or None
        Number of times to repeat the dataset.
    random_transform : bool
        Whether to apply random augmentations.
    subset : str
        'train' or 'valid'.

    Returns
    -------
    tf.data.Dataset
    """
    txp = trainpathx
    typ = tranpathy
    texp = testpathx
    teyp = testpathy
    flairp = refpath

    if subset == "train":
        lr_img_filep = os.listdir(txp)
        hr_img_filep = os.listdir(typ)
        flair_img_files = []
        if flairp is not None:
            flair_img_filep = os.listdir(flairp)

        lr_img_files = [os.path.join(txp, f) for f in lr_img_filep]
        hr_img_files = [os.path.join(typ, f) for f in hr_img_filep]
        if flairp is not None:
            flair_img_files = [os.path.join(flairp, f) for f in flair_img_filep]
    else:
        lr_img_filep = os.listdir(texp)
        hr_img_filep = os.listdir(teyp)
        flair_img_files = []
        if flairp is not None:
            flair_img_filep = os.listdir(flairp)

        lr_img_files = [os.path.join(texp, f) for f in lr_img_filep]
        hr_img_files = [os.path.join(teyp, f) for f in hr_img_filep]
        if flairp is not None:
            flair_img_files = [
                os.path.join(flairp, f) for f in flair_img_filep[1312:]
            ]

    hr_ds = tf.data.Dataset.from_tensor_slices(hr_img_files)
    hr_ds = hr_ds.map(tf.io.read_file)
    hr_ds = hr_ds.map(lambda x: tf.image.decode_png(x), num_parallel_calls=AUTOTUNE)

    lr_ds = tf.data.Dataset.from_tensor_slices(lr_img_files)
    lr_ds = lr_ds.map(tf.io.read_file)
    lr_ds = lr_ds.map(lambda x: tf.image.decode_png(x), num_parallel_calls=AUTOTUNE)

    if flairp is not None:
        flair_ds = tf.data.Dataset.from_tensor_slices(flair_img_files)
        flair_ds = flair_ds.map(tf.io.read_file)
        flair_ds = flair_ds.map(
            lambda x: tf.image.decode_png(x), num_parallel_calls=AUTOTUNE
        )
        ds = tf.data.Dataset.zip((lr_ds, hr_ds, flair_ds))
    else:
        ds = tf.data.Dataset.zip((lr_ds, hr_ds))

    ds = ds.batch(batch_size)
    ds = ds.repeat(repeat_count)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def random_crop(lr_img, hr_img, hr_crop_size=96, scale=2):
    """Random crop augmentation for paired images."""
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(
        shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32
    )
    lr_h = tf.random.uniform(
        shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32
    )

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def random_flip(lr_img, hr_img):
    """Random horizontal flip augmentation."""
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(
        rn < 0.5,
        lambda: (lr_img, hr_img),
        lambda: (tf.image.flip_left_right(lr_img),
                 tf.image.flip_left_right(hr_img)),
    )


def random_rotate(lr_img, hr_img):
    """Random 90-degree rotation augmentation."""
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)
