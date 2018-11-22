from agent import Agent
from puzzle import GameGrid
import sys
import time
import numpy as np

# normalize input values

episodes = 154000


if len(sys.argv) == 2:
    MODE = sys.argv[1]
else:
    MODE = 'train'

environment = GameGrid()
agent = Agent(MODE, episodes)


# five tup is (state, action, state_after, reward, terminal)
move = 0
for episode in range(episodes):

    still_playing = True
    recent_state = environment.give_recent_state()

    while still_playing:
        print(f'''
Episode: {episode}
Move: {move}
Randomness: {agent.epsilon}
Highest Prev Ep Score: {environment.final_score_prev}
              ''')
        if episode >= episodes - 2:
            time.sleep(0.5)
        action = agent.decide_move(recent_state)
        five_tup = environment.take_action(event=None, action=action)
        agent.add_to_replay_mem(np.array(five_tup))
        if five_tup[4] is True:
            still_playing = False

        if MODE != "play":
            if move % 32 == 0 and move != 0:
                agent.train_model()

        recent_state = five_tup[2]
        move += 1

    if MODE != "play":
        if episode % 100 == 0 and episode != 0:
            agent.target_model.set_weights(agent.model.get_weights())

    agent.episode_num += 1
    if agent.epsilon < 0 and MODE != "play":
        agent.epsilon_decay = 0
        agent.epsilon = 0.1
    agent.epsilon -= agent.epsilon_decay

print(environment.max_score, "MAX SCORE")
if MODE != "play":
    agent.save_model()
