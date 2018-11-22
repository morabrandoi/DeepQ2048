
from keras.models import Sequential, load_model
from keras.layers import Dropout, Dense, Conv2D, Flatten, MaxPooling2D
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
        self.epsilon = 1
        self.epsilon_decay = self.epsilon / (self.total_episodes-140000)

        self.num_classes = 4  # w a s d
        self.batch_size = 1
        self.epochs = 1
        self.answer_key = ["'w'", "'a'", "'s'", "'d'"]
        self.init_models()
        self.mem_capacity = 4000
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
        model.add(Conv2D(16, (2, 2), activation="relu", input_shape=(4, 4, 1)))
        model.add(Conv2D(16, (2, 2), activation="relu"))
        model.add(MaxPooling2D(pool_size=2, strides=(1, 1), padding='valid'))
        model.add(Dropout(0.25))

        model.add(Conv2D(16, (1,1), activation="relu"))
        model.add(Conv2D(16, (1,1), activation="relu"))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(16, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(self.num_classes, activation="softmax"))

        model.summary()

        model.compile(loss=self.custom_loss,
                      optimizer=Adam())

        return model

    def add_to_replay_mem(self, five_tup):
        state, action, state_after, reward, terminal = five_tup
        new_five_tup = (state, action, state_after, reward, terminal)
        self.replay_memory.append(new_five_tup)
        if len(self.replay_memory) > self.mem_capacity:
            self.replay_memory = self.replay_memory[2:]

    # def clean_state_data(self, state):
    #     print("state in clean", state)
    #     state_np = np.array(state)
    #     # state_np = state_np.flatten()
    #     return state_np

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
            train_x.append(fTup[0])

            if fTup[4] is True:
                train_y.append(fTup[3])
            else:
                processed = np.array(fTup[2]).reshape(1, 4, 4, 1)
                targ_pred = list(self.target_model.predict(processed)[0])
                target = fTup[3] + (self.gamma * max(targ_pred))
                train_y.append(target)
        train_x = np.array(train_x).reshape(32, 4, 4, 1)
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
            processed = np.array(state).reshape(1, 4, 4, 1)
            predicted_Qs = list(self.model.predict(processed)[0])
            max_q_action = max(predicted_Qs)
            action = self.answer_key[predicted_Qs.index(max_q_action)]
            return action
