#
# UNet
#

class Unet2D:

  def __init__(self, snapshot=None, n_channels=1, n_classes=2, n_levels=4,
               n_features=64, name="U-Net"):

    self.concat_blobs = []

    self.n_channels = n_channels
    self.n_classes = n_classes
    self.n_levels = n_levels
    self.n_features = n_features
    self.name = name

    self.trainModel, self.padding = self._createModel(True)
    self.testModel, _ = self._createModel(False)

    if snapshot is not None:
      self.trainModel.load_weights(snapshot)
      self.testModel.load_weights(snapshot)

  def _weighted_categorical_crossentropy(self, y_true, y_pred, weights):
    return tf.losses.softmax_cross_entropy(
      y_true, y_pred, weights=weights, reduction=tf.losses.Reduction.MEAN)

  def _createModel(self, training):

    data = keras.layers.Input(shape=(None, None, self.n_channels), name="data")

    concat_blobs = []

    if training:
      labels = keras.layers.Input(
        shape=(None, None, self.n_classes), name="labels")
      weights = keras.layers.Input(shape=(None, None), name="weights")

    # Modules of the analysis path consist of two convolutions and max pooling
    for l in range(self.n_levels):
      t = keras.layers.LeakyReLU(alpha=0.1)(
        keras.layers.Conv2D(
          2**l * self.n_features, 3, padding="valid",
          kernel_initializer="he_normal",
          name="conv_d{}a-b".format(l))(data if l == 0 else t))
      concat_blobs.append(
        keras.layers.LeakyReLU(alpha=0.1)(
          keras.layers.Conv2D(
            2**l * self.n_features, 3, padding="valid",
            kernel_initializer="he_normal", name="conv_d{}b-c".format(l))(t)))
      t = keras.layers.MaxPooling2D(pool_size=(2, 2))(concat_blobs[-1])

    # Deepest layer has two convolutions only
    t = keras.layers.LeakyReLU(alpha=0.1)(
      keras.layers.Conv2D(
        2**self.n_levels * self.n_features, 3, padding="valid",
        kernel_initializer="he_normal",
        name="conv_d{}a-b".format(self.n_levels))(t))
    t = keras.layers.LeakyReLU(alpha=0.1)(
      keras.layers.Conv2D(
        2**self.n_levels * self.n_features, 3, padding="valid",
        kernel_initializer="he_normal",
        name="conv_d{}b-c".format(self.n_levels))(t))
    pad = 8

    # Modules in the synthesis path consist of up-convolution,
    # concatenation and two convolutions
    for l in range(self.n_levels - 1, -1, -1):
      name = "upconv_{}{}{}_u{}a".format(
        *(("d", l+1, "c", l) if l == self.n_levels - 1 else ("u", l+1, "d", l)))
      t = keras.layers.LeakyReLU(alpha=0.1)(
        keras.layers.Conv2D(
          2**l * self.n_features, 2, padding="same",
          kernel_initializer="he_normal", name=name)(
            keras.layers.UpSampling2D(size = (2,2))(t)))
      t = keras.layers.Concatenate()(
        [keras.layers.Cropping2D(cropping=int(pad / 2))(concat_blobs[l]), t])
      pad = 2 * (pad + 8)
      t = keras.layers.LeakyReLU(alpha=0.1)(
        keras.layers.Conv2D(
          2**l * self.n_features, 3, padding="valid",
          kernel_initializer="he_normal", name="conv_u{}b-c".format(l))(t))
      t = keras.layers.LeakyReLU(alpha=0.1)(
        keras.layers.Conv2D(
          2**l * self.n_features, 3, padding="valid",
          kernel_initializer="he_normal", name="conv_u{}c-d".format(l))(t))
    pad /= 2

    score = keras.layers.Conv2D(
      self.n_classes, 1, kernel_initializer = 'he_normal',
      name="conv_u0d-score")(t)
    softmax_score = keras.layers.Softmax()(score)

    if training:
      model = keras.Model(inputs=[data, labels, weights], outputs=softmax_score)
      model.add_loss(
        self._weighted_categorical_crossentropy(labels, score, weights))
      adam = keras.optimizers.Adam(
        lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
        amsgrad=False)
      model.compile(optimizer=adam, loss=None)
    else:
      model = keras.Model(inputs=data, outputs=softmax_score)

    return model, int(pad)

  def loadCaffeModelH5(self, path):
    train_layer_dict = dict([(layer.name, layer)
                             for layer in self.trainModel.layers])
    test_layer_dict = dict([(layer.name, layer)
                            for layer in self.testModel.layers])
    pre = h5py.File(path, 'a')
    l = list(pre['data'].keys())
    for i in l:
      kernel = pre['data'][i]['0'][()]
      bias = pre['data'][i]['1'][()]
      train_layer_dict[i].set_weights([kernel,bias])
      test_layer_dict[i].set_weights([kernel,bias])
    pre.close()

  def train(self, sample_generator, validation_generator=None,
            n_epochs=100, snapshot_interval=1, snapshot_prefix=None):

    callbacks = [TensorBoard(log_dir="logs/{}-{}".format(self.name, time()))]
    if snapshot_prefix is not None:
      callbacks.append(keras.callbacks.ModelCheckpoint(
        (snapshot_prefix if snapshot_prefix is not None else self.name) +
        ".{epoch:04d}.h5", mode='auto', period=snapshot_interval))
    self.trainModel.fit_generator(
      sample_generator, epochs=n_epochs, validation_data=validation_generator,
      verbose=1, callbacks=callbacks)

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

