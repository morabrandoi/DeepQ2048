from agent import Agent
from puzzle import GameGrid
import sys
import time


if len(sys.argv) == 2:
    MODE = sys.argv[1]
else:
    MODE = 'play'

environment = GameGrid()
agent = Agent(MODE)

game_num = 10
# five tup is (state, action, state_after, reward, terminal)
for game in range(game_num):
    still_playing = True
    recent_state = agent.clean_state_data(environment.give_recent_state())

    while still_playing:
        # time.sleep(0.1)
        action = agent.decide_move(recent_state)

        five_tup = environment.take_action(event=None, action=action)

        environment.update_idletasks()
        environment.update()

        if five_tup[4] is True or five_tup[0] == five_tup[2]:
            still_playing = False

        recent_state = agent.clean_state_data(five_tup[2])
