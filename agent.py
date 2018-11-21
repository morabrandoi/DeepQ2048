
from keras.models import Sequential, load_model
from keras.layers import Dropout, Dense
from keras.optimizers import Adam
import random
import numpy as np


class Agent:
    def __init__(self, MODE, EPIS):
        self.mode = MODE

        self.model_name = "model.hdf5"
        self.target_model_name = "target_model.hdf5"
        self.model_file_path = f"./{self.model_name}"
        self.target_model_file_path = f"./{self.target_model_name}"

        self.total_episodes = EPIS
        self.episode_num = 0
        self.gamma = 0.99
        self.epsilon = 0.8
        self.epsilon_decay = self.epsilon / (self.total_episodes-3)

        self.num_classes = 4  # w a s d
        self.batch_size = 1
        self.epochs = 1
        self.answer_key = ["'w'", "'a'", "'s'", "'d'"]
        self.init_models()
        self.replay_memory = []

    def init_models(self):
        if self.mode == "train":
            print("CHOO CHOO")

            self.model = load_model(self.model_file_path,
                                    custom_objects={'custom_loss': self.custom_loss})

            self.target_model = load_model(self.target_model_file_path,
                                           custom_objects={'custom_loss': self.custom_loss})

        elif self.mode == "play":
            print("play play")
            self.epsilon = 0
            self.model = load_model(self.model_file_path,
                                    custom_objects={'custom_loss': self.custom_loss})

        elif self.mode == "create_new":
            print("creating new model")

            self.model = self.create_model()
            self.target_model = self.create_model()
        else:
            raise ValueError("Only use play, train, or create_new in terminal")

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

        model.compile(loss=self.custom_loss,
                      optimizer=Adam())

        return model

    def add_to_replay_mem(self, five_tup):

        state, action, state_after, reward, terminal = five_tup
        state = self.clean_state_data(state)
        state_after = self.clean_state_data(state_after)
        new_five_tup = (state, action, state_after, reward, terminal)
        self.replay_memory.append(new_five_tup)

    def clean_state_data(self, state):
        state_np = np.array(state)
        state_np = state_np.flatten()
        return state_np

    def custom_loss(self, y_pred, y_true):
        squared_error = (y_pred - y_true) ** 2
        sum_squared_error = np.sum(squared_error)
        mse = sum_squared_error / np.size(y_true)
        return(mse)

    def save_model(self):
        self.model.save(self.model_name)
        self.target_model.save(self.target_model_name)

    def train_model(self, five_tup):
        state, action, state_after, reward, terminal = five_tup
        state = self.clean_state_data(state)
        state_after = self.clean_state_data(state_after)

        if terminal is True:
            target = reward
        else:
            targ_pred = list(self.target_model.predict(np.array([state_after, ]))[0])
            target = reward + (self.gamma * max(targ_pred))

        self.history = self.model.fit(np.array([state, ]), np.array([target, ]),
                                      batch_size=self.batch_size,
                                      epochs=self.epochs,
                                      verbose=1)

        if self.episode_num % 50 == 0:
            self.save_model()

    def decide_move(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.answer_key)
        else:
            predicted_Qs = list(self.model.predict(np.array([state, ]))[0])
            max_q_action = max(predicted_Qs)
            action = self.answer_key[predicted_Qs.index(max_q_action)]
            return action
