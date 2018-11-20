from agent import Agent
from puzzle import GameGrid
import sys

if len(sys.argv) == 2:
    MODE = sys.argv[1]
else:
    MODE = 'play'

environment = GameGrid()
agent = Agent(MODE)

game_num = 10

for game in range(game_num):
    still_playing = True
    last_state = environment.give_state()
    while still_playing:
        agent.take_action(last_state)
