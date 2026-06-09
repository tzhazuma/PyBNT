"""EDSR trainer with checkpoint management."""
import tensorflow as tf
from tensorflow.python.keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.python.keras.metrics import Mean
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import (
    PiecewiseConstantDecay,
)

from pybnt.processing.models.common import evaluate


class Trainer:
    """Base trainer with checkpoint management."""

    def __init__(self, model, loss, learning_rate, checkpoint_dir='./ckpt/edsr'):
        self.now = None
        self.loss = loss
        self.checkpoint = tf.train.Checkpoint(
            step=tf.Variable(0),
            psnr=tf.Variable(-1.0),
            optimizer=Adam(learning_rate),
            model=model,
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=checkpoint_dir,
            max_to_keep=3,
        )
        self.restore()
        self.m = model

    @property
    def model(self):
        return self.checkpoint.model

    def train(self, train_dataset, valid_dataset, steps,
              evaluate_every=1000, save_best_only=False):
        """Train the model.

        Parameters
        ----------
        train_dataset : tf.data.Dataset
            Training data.
        valid_dataset : tf.data.Dataset
            Validation data.
        steps : int
            Total training steps.
        evaluate_every : int
            Evaluation frequency in steps.
        save_best_only : bool
            Only save checkpoint when PSNR improves.
        """
        loss_mean = Mean()
        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint
        vis_list = []

        for img in train_dataset.take(steps - ckpt.step.numpy()):
            if len(img) == 2:
                lr, hr = img
                ref = None
            elif len(img) == 3:
                lr, hr, ref = img
            else:
                raise ValueError("Unexpected dataset tuple length")

            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()

            loss = self.train_step(lr, hr, ref)
            loss_mean(loss)

            if step % evaluate_every == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_states()

                psnr_value = self.evaluate(valid_dataset)
                print(
                    f'{step}/{steps}: loss = {loss_value.numpy():.3f}, '
                    f'PSNR = {psnr_value.numpy():3f}'
                )
                vis_list.append((step, loss_value, psnr_value))

                if save_best_only and psnr_value <= ckpt.psnr:
                    continue

                ckpt.psnr = psnr_value
                ckpt_mgr.save()

        csv = open('./visLoss.csv', 'w')
        csv.write('step, loss, psnr\n')
        for vals in vis_list:
            csv.write('{},{},{}\n'.format(vals[0], vals[1], vals[2]))
        csv.close()

    @tf.function
    def train_step(self, lr, hr, ref=None):
        """Single training step with gradient tape."""
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)
            if ref is not None:
                ref = tf.cast(ref, tf.float32)
                sr = self.checkpoint.model([lr, ref], training=True)
            else:
                sr = self.checkpoint.model(lr, training=True)

            loss_value = self.loss(hr, sr)

        gradients = tape.gradient(
            loss_value, self.checkpoint.model.trainable_variables
        )
        self.checkpoint.optimizer.apply_gradients(
            zip(gradients, self.checkpoint.model.trainable_variables)
        )
        return loss_value

    def evaluate(self, dataset):
        """Evaluate model on a dataset."""
        return evaluate(self.checkpoint.model, dataset)

    def restore(self):
        """Restore from latest checkpoint if available."""
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)


class EdsrTrainer(Trainer):
    """EDSR-specific trainer with default hyperparameters."""

    def __init__(self, model, loss, checkpoint_dir,
                 learning_rate=None):
        if learning_rate is None:
            learning_rate = PiecewiseConstantDecay(
                boundaries=[200000], values=[1e-4, 5e-5]
            )
        if loss == 'MAE':
            loss = MeanAbsoluteError()
        elif loss == 'MSE':
            loss = MeanSquaredError()
        else:
            raise ValueError("loss specified incorrectly")
        super().__init__(
            model,
            loss=MeanAbsoluteError(),
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_dir,
        )

    def train(self, train_dataset, valid_dataset, steps=300000,
              evaluate_every=1000, save_best_only=True):
        super().train(
            train_dataset, valid_dataset, steps,
            evaluate_every, save_best_only,
        )
