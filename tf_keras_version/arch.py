import tensorflow as tf # Tensorflow 2
def cnn():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5),
                                    activation='relu',
                                    kernel_initializer='he_normal',
                                    input_shape=(512,512, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(rate = 0.2))

    model.add(tf.keras.layers.Conv2D(16, (3, 3),
                                    kernel_initializer='he_normal',
                                    activation='relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64,
                                    kernel_initializer='he_normal',
                                    activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation = 'relu'))
    return model