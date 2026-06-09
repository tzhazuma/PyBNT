"""Super-resolution inference and training API for brain images.

Uses EDSR architecture with optional Combi (dual-input) variant.
"""
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import tensorflow as tf

from pybnt.core.logconf import logger

from pybnt.processing.models.edsr import edsr, combi
from pybnt.processing.models.dataset import dataset
from pybnt.processing.models.train import EdsrTrainer
from pybnt.processing.models.common import resolve
from pybnt.processing.models import utils

# Default model parameters
DEPTH = 4
SCALE = 2
FILTERS = 64
LOSS = 'MAE'


def train_superres(
    img_dir: str,
    save_dir: str,
    label_dir: str,
    ref_dir: str | None = None,
    scale: int = 2,
    depth: int = 4,
    filters: int = 64,
    loss: str = 'MAE',
):
    """Train EDSR model for brain image super-resolution.

    Parameters
    ----------
    img_dir : str
        Path to directory containing low-resolution images.
    save_dir : str
        Path to save model checkpoints.
    label_dir : str
        Path to directory containing high-resolution labels.
    ref_dir : str or None
        Optional path to reference (flair) images for Combi model.
    scale : int
        Upscaling factor (default 2).
    depth : int
        Number of residual blocks (default 4).
    filters : int
        Number of convolution filters (default 64).
    loss : str
        Loss function: 'MAE' or 'MSE'.
    """
    train_path = os.path.join(img_dir, "train")
    val_path = os.path.join(img_dir, "val")
    utils.split_train_test(img_dir, train_path, val_path)

    label_train_path = os.path.join(label_dir, "train")
    label_val_path = os.path.join(label_dir, "val")
    utils.split_train_test(label_dir, label_train_path, label_val_path)

    train_ds = dataset(
        train_path, label_train_path,
        val_path, label_val_path,
        ref_dir,
        batch_size=16, repeat_count=1, random_transform=True,
    )
    valid_ds = dataset(
        train_path, label_train_path,
        val_path, label_val_path,
        ref_dir,
        batch_size=1, repeat_count=1, random_transform=False, subset='valid',
    )

    if ref_dir is None:
        model = edsr(scale=scale, num_filters=filters, num_res_blocks=depth,
                     res_block_scaling=0.1)
    else:
        model = combi(scale=scale, num_filters=filters, num_res_blocks=depth,
                      res_block_scaling=0.1)

    os.makedirs(f"{save_dir}/checkpoint", exist_ok=True)
    trainer = EdsrTrainer(model=model, loss=loss,
                          checkpoint_dir=f"{save_dir}/checkpoint")
    trainer.checkpoint.save(f"{save_dir}/checkpoint/1.ckpt")

    with tf.device("/gpu:0"):
        trainer.train(train_ds, valid_ds.take(10), steps=15000,
                      evaluate_every=500, save_best_only=True)

    psnrv = trainer.evaluate(valid_ds)
    logger.info(f'PSNR = {psnrv.numpy():3f}')
    trainer.checkpoint.write(f"{save_dir}/checkpoint/final.ckpt")


def run_superres(
    data_dir: str,
    checkpoint_dir: str,
    output_dir: str,
    ref_dir: str | None = None,
    scale: int = 2,
    depth: int = 4,
    filters: int = 64,
):
    """Run super-resolution inference on images.

    Parameters
    ----------
    data_dir : str
        Path to directory containing images to super-resolve.
    checkpoint_dir : str
        Path to model checkpoint directory.
    output_dir : str
        Path to save super-resolved outputs.
    ref_dir : str or None
        Optional reference image directory for Combi model.
    scale : int
        Upscaling factor.
    depth : int
        Residual blocks in model.
    filters : int
        Convolution filters in model.
    """
    if ref_dir is None:
        model = edsr(scale=scale, num_filters=filters, num_res_blocks=depth,
                     res_block_scaling=0.1)
    else:
        model = combi(scale=scale, num_filters=filters, num_res_blocks=depth,
                      res_block_scaling=0.1)

    trainer = EdsrTrainer(model=model, loss='MAE',
                          checkpoint_dir=f"{checkpoint_dir}/checkpoint")
    trainer.checkpoint.read(f"{checkpoint_dir}/checkpoint/final.ckpt")

    os.makedirs(f"{output_dir}/output", exist_ok=True)

    ref_files = []
    if ref_dir is not None:
        ref_files = sorted(os.listdir(ref_dir))

    for index, imgpath in enumerate(sorted(os.listdir(data_dir))):
        img = cv2.imread(os.path.join(data_dir, imgpath))
        img = tf.cast(img, tf.float32)
        if ref_dir is not None:
            refimg = cv2.imread(os.path.join(ref_dir, ref_files[index]))
            refimg = tf.cast(refimg, tf.float32)
            img = trainer.checkpoint.model([img, refimg], training=False)
        else:
            img = trainer.checkpoint.model(img, training=False)
        cv2.imwrite(f"{output_dir}/output/{index}.png", img.numpy())


def superres_single_image(
    image: np.ndarray,
    model=None,
    scale: int = 2,
    depth: int = 4,
    filters: int = 64,
) -> np.ndarray:
    """Apply super-resolution to a single image.

    Parameters
    ----------
    image : np.ndarray
        Input low-resolution image.
    model : tf.keras.Model or None
        Pre-loaded model. If None, creates a new EDSR model.
    scale : int
        Upscaling factor.
    depth : int
        Residual blocks.
    filters : int
        Convolution filters.

    Returns
    -------
    np.ndarray
        Super-resolved image.
    """
    if model is None:
        model = edsr(scale=scale, num_filters=filters, num_res_blocks=depth,
                     res_block_scaling=0.1)

    img = tf.cast(image, tf.float32)
    result = model(img, training=False)
    return result.numpy()
