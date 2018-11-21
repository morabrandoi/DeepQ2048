from tkinter import Frame, Label, CENTER
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

KEY_UP_ALT = "\'\\uf700\'"
KEY_DOWN_ALT = "\'\\uf701\'"
KEY_LEFT_ALT = "\'\\uf702\'"
KEY_RIGHT_ALT = "\'\\uf703\'"

KEY_UP = "'w'"
KEY_DOWN = "'s'"
KEY_LEFT = "'a'"
KEY_RIGHT = "'d'"

class GameGrid(Frame):
    def __init__(self):
        Frame.__init__(self)

        self.grid()
        self.master.title('DeepQ2048')
        self.master.bind("<Key>", self.take_action)

        self.commands = {KEY_UP: up, KEY_DOWN: down, KEY_LEFT: left,
                         KEY_RIGHT: right, KEY_UP_ALT: up, KEY_DOWN_ALT: down,
                         KEY_LEFT_ALT: left, KEY_RIGHT_ALT: right}

        self.grid_cells = []
        self.score = 0

        self.init_grid()
        self.init_matrix()
        self.update_grid_cells()
        self.update_idletasks()
        self.update()

    def init_grid(self):
        background = Frame(self, bg=BACKGROUND_COLOR_GAME,
                           width=SIZE, height=SIZE)
        background.grid()
        for i in range(GRID_LEN):
            grid_row = []
            for j in range(GRID_LEN):
                cell = Frame(background, bg=BACKGROUND_COLOR_CELL_EMPTY,
                             width=SIZE/GRID_LEN, height=SIZE/GRID_LEN)
                cell.grid(row=i, column=j,
                          padx=GRID_PADDING, pady=GRID_PADDING)
                t = Label(master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY,
                          justify=CENTER, font=FONT, width=4, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def gen(self):
        return randint(0, GRID_LEN - 1)

    def init_matrix(self):
        self.matrix = new_game(GRID_LEN)

        self.matrix = add_two_or_four(self.matrix)
        self.matrix = add_two_or_four(self.matrix)

    def update_grid_cells(self):
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(new_number), bg=BACKGROUND_COLOR_DICT[new_number], fg=CELL_COLOR_DICT[new_number])
        self.update_idletasks()

    def give_recent_state(self):
        return self.matrix

    def take_action(self, event=None, action=None):
        state, action, state_after, reward, terminal = (None, action, None, None, False)
        if event is None:
            key = action
        elif action is None:
            key = repr(event.char)

        if key in self.commands:
            state = self.matrix[:]
            self.matrix, done, score_increase = self.commands[key](self.matrix)
            # if done:
            action = key[:]
            reward = score_increase
            self.score += score_increase
            if [0 for row in self.matrix if 0 in row]:
                self.matrix = add_two_or_four(self.matrix)
            self.update_grid_cells()
            state_after = self.matrix[:]
            # done = False

            if game_state(self.matrix) == 'lose' or state_after == state:
                terminal = True
                print(f"This EP Score: {self.score}")
                print("\n\n\n\nRESET\n\n\n\n")
                self.reset_episode()
            five_tup = (state, action, state_after, reward, terminal)
            return five_tup

    def reset_episode(self):
        self.init_matrix()
        self.update_grid_cells()
        self.score = 0
