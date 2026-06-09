"""Common utilities for super-resolution: normalization, PSNR, pixel shuffle."""
import os

import cv2
import numpy as np
import tensorflow as tf

MRI_INTENSITY_MEAN = np.array([0.01212223, 0.01212223, 0.01212223])


def resolve_single(model, lr):
    """Apply super-resolution to a single image.

    Parameters
    ----------
    model : tf.keras.Model
        Super-resolution model.
    lr : tf.Tensor
        Low-resolution input image.

    Returns
    -------
    tf.Tensor
        Super-resolved image.
    """
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def resolve(model, lr_batch, ref_batch=None):
    """Apply super-resolution to a batch of images.

    Parameters
    ----------
    model : tf.keras.Model
        Super-resolution model.
    lr_batch : tf.Tensor
        Batch of low-resolution images.
    ref_batch : tf.Tensor or None
        Optional reference/flair batch.

    Returns
    -------
    tf.Tensor
        Super-resolved image batch (uint8).
    """
    lr_batch = tf.cast(lr_batch, tf.float32)
    if ref_batch is not None:
        ref_batch = tf.cast(ref_batch, tf.float32)
        sr_batch = model([lr_batch, ref_batch])
    else:
        sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch


def evaluate(model, dataset):
    """Evaluate PSNR on a dataset.

    Parameters
    ----------
    model : tf.keras.Model
        Super-resolution model.
    dataset : tf.data.Dataset
        Evaluation dataset.

    Returns
    -------
    tf.Tensor
        Mean PSNR value.
    """
    psnr_values = []
    os.makedirs("/mnt/d/bmeproject/testoutput2", exist_ok=True)
    i = 0
    for img in dataset:
        if len(img) == 3:
            lr, hr, flair = img
            sr = resolve(model, lr, flair)
        else:
            lr, hr = img
            sr = resolve(model, lr)

        batch_size = sr.shape[0]
        for j in range(batch_size):
            cv2.imwrite(
                f"/mnt/d/bmeproject/testoutput2/sr{i}_{j}.png",
                sr[j].numpy(),
            )
        psnr_value = _psnr(hr, sr)[0]
        psnr_values.append(psnr_value)
        i += 1
    return tf.reduce_mean(psnr_values)


def load_image(path):
    """Load image from path as numpy array."""
    from PIL import Image
    return np.array(Image.open(path))


def plot_sample(lr, sr):
    """Plot low-res and super-res side by side."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 10))
    images = [lr, sr]
    titles = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]})']
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])


def normalize(x, int_mean=MRI_INTENSITY_MEAN):
    """Normalize to [-1, 1] range using MRI intensity mean."""
    return (x - int_mean) / 127.5


def denormalize(x, int_mean=MRI_INTENSITY_MEAN):
    """Inverse of normalize."""
    return x * 127.5 + int_mean


def normalize_01(x):
    """Normalize images to [0, 1]."""
    return x / 255.0


def normalize_m11(x):
    """Normalize images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5


def _psnr(x1, x2):
    """Compute PSNR between two images."""
    return tf.image.psnr(x1, x2, max_val=255)


def pixel_shuffle(scale):
    """Pixel shuffle (depth-to-space) layer factory."""
    return lambda x: tf.nn.depth_to_space(x, scale)
