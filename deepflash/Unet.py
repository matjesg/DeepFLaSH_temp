import keras
from keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, MaxPooling2D, Concatenate, Dropout, BatchNormalization, Dropout, Cropping2D, UpSampling2D
import keras.optimizers
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
import os
from tqdm import tqdm
from time import time
from . import metrics
from .callbacks import CyclicLR


class Unet2D:
    def __init__(self, snapshot=None, n_channels=1, n_classes=2, n_levels=4,
                 n_features=64, batch_norm=False, relu_alpha=0.1,
                 upsample=False, k_init="he_normal", name="U-Net"):

        self.concat_blobs = []

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_levels = n_levels
        self.n_features = n_features
        self.batch_norm = batch_norm
        self.relu_alpha = relu_alpha
        self.k_init = k_init
        self.upsample = upsample
        self.name = name
        self.metrics = [metrics.recall,
                        metrics.precision,
                        metrics.f1,
                        metrics.competitionMetric2,
                        metrics.mcor]

        self.trainModel, self.padding = self._createModel(True)
        self.testModel, _ = self._createModel(False)

        if snapshot is not None:
            self.trainModel.load_weights(snapshot)
            self.testModel.load_weights(snapshot)

    def _weighted_categorical_crossentropy(self, y_true, y_pred, weights):
        return tf.losses.softmax_cross_entropy(y_true, y_pred, weights=weights, reduction=tf.losses.Reduction.MEAN)

    def _createModel(self, training):

        data = keras.layers.Input(shape=(None, None, self.n_channels), name="data")

        concat_blobs = []

        if training:
            labels = keras.layers.Input(
                shape=(None, None, self.n_classes), name="labels")
            weights = keras.layers.Input(shape=(None, None), name="weights")

        # Modules of the analysis path consist of two convolutions and max pooling
        for l in range(self.n_levels):
            t = Conv2D(2**l * self.n_features, 3, padding="valid", kernel_initializer=self.k_init,
                       name="conv_d{}a-b".format(l))(data if l == 0 else t)
            t = LeakyReLU(alpha=self.relu_alpha)(t)
            if self.batch_norm:
                t = BatchNormalization(
                    axis=-1, momentum=0.99, epsilon=0.001)(t)
            t = Conv2D(2**l * self.n_features, 3, padding="valid",
                       kernel_initializer=self.k_init, name="conv_d{}b-c".format(l))(t)
            t = LeakyReLU(alpha=self.relu_alpha)(t)
            if self.batch_norm:
                t = BatchNormalization(
                    axis=-1, momentum=0.99, epsilon=0.001)(t)
            # if l >= 2:
            #    t = Dropout(rate=0.5)(t)
            concat_blobs.append(t)
            t = keras.layers.MaxPooling2D(pool_size=(2, 2))(concat_blobs[-1])

        # Deepest layer has two convolutions only
        t = Conv2D(2**self.n_levels * self.n_features, 3, padding="valid",
                   kernel_initializer=self.k_init, name="conv_d{}a-b".format(self.n_levels))(t)
        t = LeakyReLU(alpha=self.relu_alpha)(t)
        t = Conv2D(2**self.n_levels * self.n_features, 3, padding="valid",
                   kernel_initializer=self.k_init, name="conv_d{}b-c".format(self.n_levels))(t)
        t = LeakyReLU(alpha=self.relu_alpha)(t)
        pad = 8

        # Modules in the synthesis path consist of up-convolution,
        # concatenation and two convolutions
        for l in range(self.n_levels - 1, -1, -1):
            name = "upconv_{}{}{}_u{}a".format(
                *(("d", l+1, "c", l) if l == self.n_levels - 1 else ("u", l+1, "d", l)))
            if self.upsample:
                t = UpSampling2D(size=(2, 2), name=name)(t)
            else:
                t = Conv2DTranspose(2**np.max((l, 1)) * self.n_features, (2, 2), strides=2,
                                    padding='valid', kernel_initializer=self.k_init, name=name)(t)
                t = LeakyReLU(alpha=self.relu_alpha)(t)
            t = Concatenate()(
                [Cropping2D(cropping=int(pad / 2))(concat_blobs[l]), t])

            # if self.batch_norm:
            #    t = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(t)
            t = Conv2D(2**np.max((l, 1)) * self.n_features, 3, padding="valid",
                       kernel_initializer=self.k_init, name="conv_u{}b-c".format(l))(t)
            t = LeakyReLU(alpha=self.relu_alpha)(t)
            # if self.batch_norm:
            #    t = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(t)
            t = Conv2D(2**np.max((l, 1)) * self.n_features, 3, padding="valid",
                       kernel_initializer=self.k_init, name="conv_u{}c-d".format(l))(t)
            t = LeakyReLU(alpha=self.relu_alpha)(t)
            pad = 2 * (pad + 8)

        pad /= 2

        score = Conv2D(self.n_classes, 1,
                       kernel_initializer=self.k_init, name="conv_u0d-score")(t)
        softmax_score = keras.layers.Softmax()(score)

        if training:
            model = keras.Model(
                inputs=[data, labels, weights], outputs=softmax_score)
            model.add_loss(self._weighted_categorical_crossentropy(
                labels, score, weights))
            opt = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            model.compile(optimizer=opt, loss=None)
            for m in self.metrics:
                model.metrics_tensors.append(m(labels, score))
                model.metrics_names.append(m.__name__)
            
        else:
            model = keras.Model(inputs=data, outputs=softmax_score)

        return model, int(pad)


    def train(self, sample_generator, validation_generator=None,
              n_epochs=100, snapshot_interval=1, snapshot_prefix=None,
             cyclic_lr= None):

        callbacks = [TensorBoard(
            log_dir="logs/{}-{}".format(self.name, time()))]
        if snapshot_prefix is not None:
            c_dir = 'checkpoints'
            if not os.path.isdir(c_dir):
                os.makedirs(c_dir)
            c_path = os.path.join(
                c_dir, (snapshot_prefix if snapshot_prefix is not None else self.name))
            callbacks.append(ModelCheckpoint(
                c_path + ".{epoch:04d}.h5", mode='auto', period=snapshot_interval))
        if cyclic_lr is not None:
             callbacks.append(CyclicLR(base_lr=0.00001, 
                                       max_lr=0.0001, 
                                       step_size=750., # Authors suggest setting step_size = (2-8) x (training iterations in epoch=63)
                                       mode=cyclic_lr))
                
        self.trainModel.fit_generator(sample_generator,
                                      steps_per_epoch=len(sample_generator)*9,
                                      epochs=n_epochs,
                                      validation_data=validation_generator,
                                      verbose=1,
                                      callbacks=callbacks)

    def predict(self, tile_generator):

        smscores = []
        segmentations = []

        for tileIdx in range(tile_generator.__len__()):
            tile = tile_generator.__getitem__(tileIdx)
            outIdx = tile[0]["image_index"]
            outShape = tile[0]["image_shape"]
            outSlice = tile[0]["out_slice"]
            inSlice = tile[0]["in_slice"]
            softmax_score = self.testModel.predict(tile[0]["data"], verbose=1)
            if len(smscores) < outIdx + 1:
                smscores.append(np.empty((*outShape, self.n_classes)))
                segmentations.append(np.empty(outShape))
            smscores[outIdx][outSlice] = softmax_score[0][inSlice]
            segmentations[outIdx][outSlice] = np.argmax(
                softmax_score[0], axis=-1)[inSlice]

        return smscores, segmentations
