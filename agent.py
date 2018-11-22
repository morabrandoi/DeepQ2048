
from keras.models import Sequential, load_model
from keras.layers import Dropout, Dense
from keras.optimizers import Adam
import random
import numpy as np
import pickle


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
        self.epsilon = 1.0
        self.epsilon_decay = self.epsilon / (self.total_episodes-3)

        self.num_classes = 4  # w a s d
        self.batch_size = 1
        self.epochs = 1
        self.answer_key = ["'w'", "'a'", "'s'", "'d'"]
        self.init_models()
        self.mem_capacity = 1000
        self.replay_memory_file = "replay_memory.p"
        self.replay_memory = self.init_replay_mem()

    def init_replay_mem(self):
        try:
            return pickle.load(open(self.replay_memory_file, "rb"))
        except:
            return []

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
        if len(self.replay_memory) > self.mem_capacity:
            self.replay_memory = self.replay_memory[2:]

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
        pickle.dump(self.replay_memory, open(self.replay_memory_file, "wb"))

    def train_model(self):
        sample = random.sample(self.replay_memory, 32)
        train_y = []
        train_x = []
        for fTup in sample:
            fTup = list(fTup)
            fTup[0] = self.clean_state_data(fTup[0])
            fTup[2] = self.clean_state_data(fTup[2])

            train_x.append(fTup[0])

            if fTup[4] is True:
                train_y.append(fTup[3])
            else:
                targ_pred = list(self.target_model.predict(np.array([fTup[2], ]))[0])
                target = fTup[3] + (self.gamma * max(targ_pred))
                train_y.append(target)
        train_x = np.array(train_x)
        train_y = np.array(train_y)

        self.history = self.model.fit(train_x, train_y,
                                      batch_size=self.batch_size,
                                      epochs=self.epochs,
                                      verbose=1)

        if self.episode_num % 50 == 0 and self.episode_num != 0:
            self.save_model()

    def decide_move(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.answer_key)
        else:
            predicted_Qs = list(self.model.predict(np.array([state, ]))[0])
            max_q_action = max(predicted_Qs)
            action = self.answer_key[predicted_Qs.index(max_q_action)]
            return action
