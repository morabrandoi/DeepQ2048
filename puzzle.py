# from tkinter import Frame, Label, CENTER
from logic import new_game, add_two_or_four, game_state, up, down, left, right
from random import randint


SIZE = 500
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
                         16: "#f59563", 32: "#f67c5f", 64: "#f65e3b",
                         128: "#edcf72", 256: "#edcc61", 512: "#edc850",
                         1024: "#edc53f", 2048: "#edc22e", 4096: "#559938",
                         8192: "#86e7f4"}

CELL_COLOR_DICT = {2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
                   32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2",
                   256: "#f9f6f2", 512: "#f9f6f2", 1024: "#f9f6f2",
                   2048: "#f9f6f2", 4096: "#f9f6f2", 8192: "#776e65"}
FONT = ("Verdana", 40, "bold")


KEY_UP = "'w'"
KEY_DOWN = "'s'"
KEY_LEFT = "'a'"
KEY_RIGHT = "'d'"


class GameGrid():
    def __init__(self):

        self.commands = {KEY_UP: up, KEY_DOWN: down, KEY_LEFT: left,
                         KEY_RIGHT: right}

        self.final_score_prev = -1
        self.score = 0
        self.max_score = 0
        self.init_matrix()

    def gen(self):
        return randint(0, GRID_LEN - 1)

    def init_matrix(self):
        self.matrix = new_game(GRID_LEN)

        self.matrix = add_two_or_four(self.matrix)
        self.matrix = add_two_or_four(self.matrix)

    def give_recent_state(self):
        return self.matrix

    def take_action(self, event=None, action=None):
        generated_new = False
        state, action, state_after, reward, terminal = (None, action, None, None, False)
        if event is None:
            key = action
        elif action is None:
            key = repr(event.char)

        if key in self.commands:
            state = self.matrix[:]
            self.matrix, done, score_increase = self.commands[key](self.matrix)

            action = key[:]
            reward = score_increase
            self.score += score_increase
            if [0 for row in self.matrix if 0 in row]:
                generated_new = True
                self.matrix = add_two_or_four(self.matrix)
            state_after = self.matrix[:]

            if game_state(self.matrix) == 'lose' or (state_after == state and generated_new == False):
                reward -= 100
                terminal = True
                print(f"This EP Score: {self.score}")
                self.reset_episode()
            five_tup = (state, action, state_after, reward, terminal)
            [print(row) for row in state]
            print(action)
            return five_tup

    def reset_episode(self):
        self.init_matrix()
        self.final_score_prev = self.score
        self.max_score = max(self.max_score, self.score)
        self.score = 0
