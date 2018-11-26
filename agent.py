
from keras.models import Sequential, load_model
from keras.layers import Dropout, Dense, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import Adam
import random
import numpy as np
import pickle


class Agent:
    def __init__(self, MODE, EPISODES):
        self.mode = MODE

        self.model_name = "model.hdf5"
        self.target_model_name = "target_model.hdf5"
        self.model_file_path = f"./{self.model_name}"
        self.target_model_file_path = f"./{self.target_model_name}"
        self.learning_rate = 0.01

        self.num_classes = 4  # w a s d
        self.answer_key = ["'w'", "'a'", "'s'", "'d'"]
        self.mem_capacity = 4000
        self.replay_memory_file = "replay_memory.p"
        self.replay_memory = self.init_replay_mem()

        self.total_episodes = EPISODES
        self.episode_num = 0
        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / 1000

        self.init_models()

    def init_replay_mem(self):
        try:
            return pickle.load(open(self.replay_memory_file, "rb"))
        except:
            return []

    def init_models(self):
        if self.mode == "train":
            self.model = load_model(self.model_file_path)
            self.target_model = load_model(self.target_model_file_path)

        elif self.mode == "play":
            self.epsilon = 0
            self.model = load_model(self.model_file_path)

        elif self.mode == "create_new":
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

        model.add(Conv2D(16, (1, 1), activation="relu"))
        model.add(Conv2D(16, (1, 1), activation="relu"))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(16, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(self.num_classes, activation="softmax"))

        model.summary()

        model.compile(loss="mse",
                      optimizer=Adam(lr=self.learning_rate))

        return model

    def remember(self, five_tup):
        self.replay_memory.append(five_tup)
        if len(self.replay_memory) > self.mem_capacity:
            self.replay_memory = self.replay_memory[2:]

    def train_model(self, batch_size=50):
        minibatch = random.sample(self.replay_memory, batch_size)
        train_y = []
        train_x = []
        for fTup in minibatch:
            state_before_action, action, state_after_action, reward, done = list(fTup)
            action_index = self.answer_key.index(action)
            train_x.append(state_before_action)

            if done is True:
                target_for_one_action = reward
            else:
                processed_state_after = np.array(state_after_action).reshape(1, 4, 4, 1)
                target_for_one_action = reward + (self.gamma * np.amax(self.target_model.predict(processed_state_after)[0]))

            processed_state_before = np.array(state_before_action).reshape(1,4,4,1)
            target_list = self.model.predict(processed_state_before)[0]
            target_list[action_index] = target_for_one_action
            train_y.append(target_list)

        train_x = np.array(train_x).reshape(batch_size, 4, 4, 1)
        train_y = np.array(train_y)
        self.history = self.model.fit(train_x, train_y,
                                      epochs=1,
                                      verbose=1)

        if self.episode_num % 25 == 0 and self.episode_num != 0:
            self.save_model()

    def save_model(self):
        self.model.save(self.model_name)
        self.target_model.save(self.target_model_name)
        pickle.dump(self.replay_memory, open(self.replay_memory_file, "wb"))

    def decide_move(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.answer_key)
        else:
            processed_state = np.array(state).reshape(1, 4, 4, 1)
            action = self.answer_key[np.argmax(self.model.predict(processed_state)[0])]
            return action

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
