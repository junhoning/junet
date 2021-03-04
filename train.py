import os
from glob import glob
from datetime import datetime

import numpy as np
from tqdm import notebook

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

import junet
from junet import models
from junet.optimizers import losses, metrics
from junet.data import Dataset


class Model:
    def __init__(self, 
                save_dir,
                input_shape=(128, 128, 128, 1),
                num_classes=3,
                model_name='vanilla',
                load_model=False,
                memo='',
                **kwargs):
        self.model_name = model_name.lower()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.save_dir = save_dir
        self.memo = memo
        self.save_name = f'{self.train_time}_{model_name}{memo}'
        self.num_dims = len(input_shape[:-1])

        if 'h_params' in kwargs:
            self.is_hparams = True
            self.hparams = kwargs['h_params']
            print('HParams', self.hparams)
        else:
            self.is_hparams = False
        self.model = self.get_model(model_name, load_model)

        print(f"Start with {model_name}{memo}")
    
    def get_model(self, model_name, load_model=False):
        if model_name == 'vanilla':
            model = models.vanilla_unet.get_model(self.input_shape, self.num_classes)
        elif model_name == 'xception_3d':
            model = models.xception_unet_3d.get_model(self.input_shape, self.num_classes)
        elif model_name == 'dense_3d':
            model = models.dense_unet_3d.get_model(self.input_shape, self.num_classes)
        elif model_name == 'resnet_3d':
            model = models.res_unet_3d.get_model(self.input_shape, self.num_classes)
        elif model_name == 'vanilla_3d':
            model = models.vanilla_unet_3d.get_model(self.input_shape, self.num_classes)
        else:
            raise "Please select from (vanilla, xception_3d, dense_3d, resnet_3d, vanilla_3d)"
        print(f'Model is {model_name}')
        
        # Load Model
        pretrained_models = glob(os.path.join(self.save_dir, f'*_{model_name}.h5'))

        if len(pretrained_models) > 0 and load_model:
            pretrained_models.sort()
            load_path = pretrained_models[-1]
            self.model.load_weights(load_path)
            print("Model loaded", load_path)
        
        return model
    
    def set_optm(self, optm='adam', learning_rate=0.001, lr_schedule=False):
        if lr_schedule:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate,
                decay_steps=200,
                decay_rate=0.9,
                staircase=True
                )

        if optm.lower() == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        elif optm.lower() == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate)
        else:
            raise "Wrong Optimization Name (adam, rmsprop)"

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_dice = tf.keras.metrics.Mean(name='train_dice')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_dice = tf.keras.metrics.Mean(name='test_dice')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            Losses = losses.Losses('soft_dice_loss')
            loss = Losses.loss_function(labels, predictions)
            # en_loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_dice(metrics.dice(labels, predictions))
        self.train_accuracy(labels, predictions)
        return predictions
        

    @tf.function
    def test_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        Losses = losses.Losses('soft_dice_loss')
        loss = Losses.loss_function(labels, predictions)

        self.test_loss(loss)
        self.test_dice(metrics.dice(labels, predictions))
        self.test_accuracy(labels, predictions)
        return predictions

    def get_dataset(self, train_filename, test_filename, augmentations, batch_size=1):
        # self.train_ds = Dataset(self.input_shape, self.num_classes, batch_size, augmentations, is_training=True).get_dataset(train_filename)
        # self.test_ds = Dataset(self.input_shape, self.num_classes, batch_size, augmentations, is_training=False).get_dataset(test_filename)
        self.train_ds, self.test_ds = Dataset(self.input_shape, self.num_classes, batch_size, augmentations, is_training=False).get_dataset(test_filename)
    
    def normalization(self, image):
        return (image - np.min(image)) / (np.max(image) - np.min(image))
    
    def get_sample(self):
        image, label = next(iter(self.train_ds))
        return image, label

    def fit(self, num_epochs=9999):
        # Set Log 
        save_path = os.path.join(self.save_dir, self.save_name)

        train_logdir = os.path.join('logs/train', self.save_name)
        test_logdir = os.path.join('logs/test', self.save_name)

        train_summary_writer = tf.summary.create_file_writer(train_logdir)
        test_summary_writer = tf.summary.create_file_writer(test_logdir)

        curr_loss = 1

        for epoch in range(num_epochs):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.train_dice.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()
            self.test_dice.reset_states()

            def reduce_dim(image, is_argmax=True):
                if is_argmax:
                    image = tf.argmax(image, -1)
                if self.num_dims == 3:
                    return image[0, :, :, :, tf.newaxis]
                else:
                    return image[:, :, :, tf.newaxis]

            print("Start Training : ", datetime.now(), ', @', self.save_name)
            for images, labels in self.train_ds:  # notebook.tqdm(self.train_ds):  
                preds = self.train_step(images, labels)
                
            with train_summary_writer.as_default():
                tf.summary.scalar('loss_value', self.train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', self.train_accuracy.result(), step=epoch)
                tf.summary.scalar('dice_score', self.train_dice.result(), step=epoch)

                disp_idx = np.random.randint(images.shape[1])

                tf.summary.image('input_image', reduce_dim(images, False), epoch)
                tf.summary.image('preds_image', self.normalization(reduce_dim(preds)), epoch)
                tf.summary.image('label_image', self.normalization(reduce_dim(labels)), epoch)

            for images, labels in self.train_ds:  # notebook.tqdm(self.test_ds):       
                test_preds = self.test_step(images, labels)

            with test_summary_writer.as_default():
                tf.summary.scalar('loss_value', self.test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', self.test_accuracy.result(), step=epoch)
                tf.summary.scalar('dice_score', self.test_dice.result(), step=epoch)

                disp_idx = np.random.randint(images.shape[1])
                tf.summary.image('input_image', reduce_dim(images, False), epoch)
                tf.summary.image('preds_image', self.normalization(reduce_dim(test_preds)), epoch)
                tf.summary.image('label_image', self.normalization(reduce_dim(labels)), epoch)

                if self.is_hparams:
                    hp.hparams(self.hparams)

            print(f'Epoch {epoch + 1}, Loss: {self.train_loss.result()}, Accuracy: {self.train_accuracy.result() * 100}, Dice: {self.train_dice.result() * 100}, Test Loss: {self.test_loss.result()}, Test Accuracy: {self.test_accuracy.result() * 100}, Test Dice: {self.test_dice.result() * 100}')
            
            if curr_loss > self.train_loss.result():
                curr_loss = self.train_loss.result()
                self.model.save_weights(f"{save_path}_{self.model_name}{self.memo}.h5")
                print(f"Model saved {self.save_name}")
