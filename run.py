from agent import Agent
from puzzle import GameGrid
import sys
import time

# MAKE NOT ASYNCHRONUS

episodes = 24000


if len(sys.argv) == 2:
    MODE = sys.argv[1]
else:
    MODE = 'train'

environment = GameGrid()
agent = Agent(MODE, episodes)


# five tup is (state, action, state_after, reward, terminal)
for episode in range(episodes):

    still_playing = True
    recent_state = agent.clean_state_data(environment.give_recent_state())

    move = 0
    while still_playing:
        print(f"Episode {episode} and randomness at {agent.epsilon}")
        if episode >= episodes - 2:
            time.sleep(0.5)
        action = agent.decide_move(recent_state)
        five_tup = environment.take_action(event=None, action=action)

        agent.add_to_replay_mem(five_tup)
        if five_tup[4] is True or five_tup[0] == five_tup[2]:
            still_playing = False

        if MODE != "play":
            if move % 32 == 0 and move != 0:
                agent.train_model()
            if move % 100 == 0:
                agent.target_model.set_weights(agent.model.get_weights())

        recent_state = agent.clean_state_data(five_tup[2])

        move += 1

    agent.episode_num += 1
    agent.epsilon -= agent.epsilon_decay

print(environment.max_score, "MAX SCORE")
agent.save_model()
