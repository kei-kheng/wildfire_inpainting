import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import MaxPooling2D, Conv2DTranspose, Conv2D, Concatenate
from keras.layers import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from utils import simulate_agent_on_samples
import csv

print("GPU: ", tf.config.experimental.list_physical_devices("GPU"))


# Loads the MNIST dataset.
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
        path="mnist.npz"
    )
    x_train = (x_train.astype(np.float32)) / 255.0

    x_train = x_train.reshape(60000, 28, 28, 1)
    return (x_train, y_train, x_test, y_test)


(X_train, Y_train, X_test, Y_test) = load_data()
# Get 409600 samples from the dataset
ix = np.random.randint(0, X_train.shape[0], 409600)
X_COMPLETE = X_train[ix]
X_MISSING, MASKS = simulate_agent_on_samples(X_COMPLETE)
print(X_MISSING.shape, MASKS.shape)


def create_old_encoder(DIMENSIONS):
    X = Input(shape=(28, 28, 1))

    x = Conv2D(8, kernel_size=4, padding="same", activation="relu")(X)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, kernel_size=4, padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)

    x = Dense(32, activation="relu")(x)

    x = Dense(DIMENSIONS, activation="relu")(x)

    model = Model(inputs=[X], outputs=x)
    model.compile(loss="binary_crossentropy", optimizer=Adam())

    return model


def create_encoder(DIMENSIONS):
    X = Input(shape=(28, 28, 1))
    M = Input(shape=(28, 28, 1))

    m = MaxPooling2D(pool_size=(4, 4))(M)
    m = Flatten()(m)

    x = Conv2D(8, kernel_size=4, padding="same", activation="relu")(X)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, kernel_size=4, padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)

    c = Concatenate()([x, m])

    c = Dense(64, activation="relu")(c)

    c = Dense(DIMENSIONS, activation="relu")(c)

    model = Model(inputs=[X, M], outputs=c)
    model.compile(loss="binary_crossentropy", optimizer=Adam())

    return model


def create_decoder(DIMENSIONS):
    X = Input(shape=(DIMENSIONS,))

    c = Dense(49, activation="relu")(X)
    c = Reshape((7, 7, 1))(c)
    c = Conv2DTranspose(32, kernel_size=4, padding="same", activation="relu")(c)
    c = UpSampling2D(size=(2, 2))(c)
    c = Conv2DTranspose(16, kernel_size=4, padding="same", activation="relu")(c)
    c = UpSampling2D(size=(2, 2))(c)
    c = Conv2DTranspose(1, kernel_size=4, padding="same", activation="sigmoid")(c)

    model = Model(inputs=[X], outputs=c)
    model.compile(loss="binary_crossentropy", optimizer=Adam())

    return model


def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = np.ones((n_samples, 1))
    return X, y


def generate_missing_samples(dataset, n_samples):
    # choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    X, M = simulate_agent_on_samples(X)
    return X, M, dataset[ix]


def summarize_performance(epoch, encoder, dataset, n_samples=100):
    # generate points in latent space
    x_input, masks, complete = generate_missing_samples(dataset, n_samples)
    # predict outputs
    x = np.nan_to_num(x_input, 0)
    X = encoder.predict([x, masks])

    plt.figure(figsize=(30, 10))
    # plot images
    for i in range(100):
        # define subplot
        plt.subplot(10, 30, 3 * i + 1)
        # turn off axis
        plt.axis("off")
        # plot raw pixel data
        plt.imshow(X[i])

        plt.subplot(10, 30, 3 * i + 2)
        # turn off axis
        plt.axis("off")
        # plot raw pixel data
        plt.imshow(x_input[i])

        plt.subplot(10, 30, 3 * i + 3)
        # turn off axis
        plt.axis("off")
        # plot raw pixel data
        plt.imshow(masks[i])

    # save plot to file
    filename = "generated_plot_e%03d.png" % (epoch + 1)
    plt.savefig(filename)
    plt.close()


def summarize_learning(epoch, old_pair, dataset):
    # generate points in latent space
    ix = np.random.randint(0, dataset.shape[0], 100)
    x = dataset[ix]
    # predict outputs

    X = old_pair.predict([x])

    plt.figure(figsize=(30, 10))
    # plot images
    for i in range(100):
        # define subplot
        plt.subplot(10, 30, 3 * i + 1)
        # turn off axis
        plt.axis("off")
        # plot raw pixel data
        plt.imshow(X[i])

        plt.subplot(10, 30, 3 * i + 2)
        # turn off axis
        plt.axis("off")
        # plot raw pixel data
        plt.imshow(x[i])

    # save plot to file
    filename = "first_step_generated_plot_e%03d.png" % (epoch + 1)
    plt.savefig(filename)
    plt.close()


def train(dims=20):
    # Load data
    (X_train, _, _, _) = load_data()

    # Create ecoders
    old_encoder = create_old_encoder(dims)
    # Create decoder
    decoder = create_decoder(dims)

    # Create pair of encoder/decoder
    X = Input(shape=(28, 28, 1))
    encoded = old_encoder([X])
    decoded = decoder(encoded)
    old_pair = Model([X], decoded)
    old_pair.compile(loss="binary_crossentropy", optimizer=Adam())

    loss1 = list()
    loss2 = list()

    # Train encoder/decoder pair on normal MNIST data to learn latent space
    for e in range(200):
        print("Epoch: ", e)
        ix = np.random.randint(0, X_train.shape[0], 2048)
        x = X_train[ix]

        loss = old_pair.fit([x], x, epochs=1, verbose=1)
        loss1.append(loss.history["loss"][0])

        if e % 10 == 0:
            summarize_learning(e, old_pair, X_train)

    # Throw away old encoder and train new one, so it learns to map training data to latent space
    X = Input(shape=(28, 28, 1))
    M = Input(shape=(28, 28, 1))
    encoder = create_encoder(dims)
    encoded = encoder([X, M])
    decoded = decoder(encoded)
    decoder.trainable = False
    new_pair = Model([X, M], decoded)
    new_pair.compile(loss="binary_crossentropy", optimizer=Adam())

    for e in range(800):
        print("Epoch: ", e)
        missing, masks, complete = (
            X_MISSING[e * 512 : (e + 1) * 512],
            MASKS[e * 512 : (e + 1) * 512],
            X_COMPLETE[e * 512 : (e + 1) * 512],
        )
        x = np.nan_to_num(missing, 0)

        loss = new_pair.fit([x, masks], complete, epochs=1, verbose=1)
        loss2.append(loss.history["loss"][0])

        if e % 10 == 0:
            print("Summarizing performance")
            summarize_performance(e, new_pair, X_train)

    encoder.save("models2/mask/encoder.h5")
    decoder.save("models2/mask/decoder.h5")

    with open("losses.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(["2P Mask {}".format(dims)])
        writer.writerow(loss2)


def train_alt(dims=20):
    # Load data
    (X_train, _, _, _) = load_data()

    # Create ecoders
    encoder = create_encoder(dims)
    # Create decoder
    decoder = create_decoder(dims)

    loss1 = list()

    X = Input(shape=(28, 28, 1))
    M = Input(shape=(28, 28, 1))
    encoded = encoder([X, M])
    decoded = decoder(encoded)
    new_pair = Model([X, M], decoded)
    new_pair.compile(loss="binary_crossentropy", optimizer=Adam())

    for e in range(200):
        print("Epoch: ", e)
        missing, masks, complete = (
            X_MISSING[e * 2048 : (e + 1) * 2048],
            MASKS[e * 2048 : (e + 1) * 2048],
            X_COMPLETE[e * 2048 : (e + 1) * 2048],
        )
        x = np.nan_to_num(missing, 0)

        loss = new_pair.fit([x, masks], complete, epochs=1, verbose=1)
        loss1.append(loss.history["loss"][0])

        if e % 10 == 0:
            print("Summarizing performance")
            summarize_performance(e, new_pair, X_train)

    with open("losses.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(["1P Mask {}".format(dims)])
        writer.writerow(loss1)


def train_no_mask(dims=20):
    # Load data
    (X_train, _, _, _) = load_data()

    # Create ecoders
    old_encoder = create_old_encoder(dims)
    # Create decoder
    decoder = create_decoder(dims)

    # Create pair of encoder/decoder
    X = Input(shape=(28, 28, 1))
    encoded = old_encoder([X])
    decoded = decoder(encoded)
    old_pair = Model([X], decoded)
    old_pair.compile(loss="binary_crossentropy", optimizer=Adam())

    loss1 = list()

    # Train encoder/decoder pair on normal MNIST data to learn latent space
    for e in range(200):
        print("Epoch: ", e)
        missing, masks, complete = (
            X_MISSING[e * 2048 : (e + 1) * 2048],
            MASKS[e * 2048 : (e + 1) * 2048],
            X_COMPLETE[e * 2048 : (e + 1) * 2048],
        )
        x = np.nan_to_num(missing, 0)

        loss = old_pair.fit([x], complete, epochs=1, verbose=1)
        loss1.append(loss.history["loss"][0])

        if e % 10 == 0:
            summarize_learning(e, old_pair, X_train)

    with open("losses.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(["1P No Mask {}".format(dims)])
        writer.writerow(loss1)


train(20)
train(40)
