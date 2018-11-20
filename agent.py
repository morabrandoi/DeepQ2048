from keras.models import Sequential, load_model
from keras.layers import Dropout, Dense
from keras.optimizers import Adam
import random
import numpy as np


class Agent:
    def __init__(self, MODE):
        self.mode = MODE

        self.model_name = "model.hdf5"
        self.target_model_name = "target_model.hdf5"

        self.model_file_path = f"./{self.model_name}"
        self.target_model_file_path = f"./{self.target_model_name}"

        self.num_classes = 4  # up down left right
        self.batch_size = 4
        self.epochs = 1
        self.answer_key = ["'w'", "'a'", "'s'", "'d'"]
        self.init_models()
        self.replay_memory = []

    def init_models(self):
        if self.mode == "train":
            print("CHOO CHOO")

            self.model = load_model(self.model_file_path)
            self.target_model = load_model(self.target_model_file_path)

        elif self.mode == "play":
            print("play play")

            self.model = load_model(self.model_file_path)

        elif self.mode == "create_new":
            print("creating new model")

            self.model = self.create_model()
            self.target_model = self.create_model()
        else:
            raise ValueError("Only use play or train in terminal")

    def create_model(self):
        model = Sequential()
        model.add(Dense(100, activation="relu", input_shape=(16,)))
        model.add(Dropout(0, 2))

        model.add(Dense(100, activation="relu"))
        model.add(Dropout(0, 2))

        model.add(Dense(16, activation="relu"))
        model.add(Dropout(0, 2))

        model.add(Dense(self.num_classes, activation="softmax"))

        model.summary()

        model.compile(loss="categorical_crossentropy",
                      optimizer=Adam(),
                      metrics=["accuracy"])

        return model

    def add_to_replay_mem(self, five_tup):
        if random.random() < 0.10:
            state, action, state_after, reward, terminal = five_tup
            state = self.clean_state_data(state)
            state_after = self.clean_state_data(state_after)
            new_five_tup = (state, action, state_after, reward, terminal)
            self.replay_memory.append(new_five_tup)

    def clean_state_data(self, state):
        state_np = np.array(state)
        state_np = state_np.flatten()

        return state_np

    def train_model(self, x_train):
        # five tup is state, action, state_after, reward, terminal
        target = 0
        self.history = self.model.fit(x_train, target,
                                      batch_size=self.batch_size,
                                      epochs=self.epochs,
                                      verbose=1)

        self.model.save(self.model_name)
        self.target_model.save(self.target_model_name)

    def decide_move(self, state):
        predicted_Qs = list(self.model.predict(np.array([state, ]))[0])
        max_q_action = max(predicted_Qs)
        action = self.answer_key[predicted_Qs.index(max_q_action)]
        return action

    '''
    batch_size = 512
    num_classes = 10
    epochs = 400

    # the data shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print(x_train.shape)

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    x_train /= 255
    x_test /= 255


    # conver class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    # defining model
    model = Sequential()
    model.add(Dense(400, activation="relu", input_shape=(784,)))
    model.add(Dropout(0, 2))

    model.add(Dense(200, activation="relu"))
    model.add(Dropout(0, 2))

    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0, 2))

    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0, 2))

    model.add(Dense(25, activation="relu"))
    model.add(Dropout(0, 2))

    model.add(Dense(num_classes, activation="softmax"))

    model.summary()


    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(),
                  metrics=["accuracy"])

    early_stop = keras.callbacks.EarlyStopping(monitor='val_acc',
                                               min_delta=0.01,
                                               patience=2,
                                               verbose=0,
                                               mode='auto',
                                               baseline=None)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[early_stop],
                        verbose=1,
                        validation_data=(x_test, y_test))
    # model.save("./mnistNet.hdf5")

    score = model.evaluate(x_test, y_test, verbose=1)
    print("Test loss: ", score[0])
    print("test accuracy: ", score[1])
    '''
