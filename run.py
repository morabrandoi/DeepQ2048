from agent import Agent
from puzzle import GameGrid
import sys
import time

# MAKE NOT ASYNCHRONUS

episodes = 100


if len(sys.argv) == 2:
    MODE = sys.argv[1]
else:
    MODE = 'train'

environment = GameGrid()
agent = Agent(MODE, episodes)


# five tup is (state, action, state_after, reward, terminal)
for episode in range(episodes):
    print(f"Episode {episode} and randomness at {agent.epsilon}")
    still_playing = True
    recent_state = agent.clean_state_data(environment.give_recent_state())

    while still_playing:
        time.sleep(0.05)
        if episode >= episodes - 2:
            time.sleep(0.5)
        action = agent.decide_move(recent_state)
        five_tup = environment.take_action(event=None, action=action)

        environment.update_idletasks()
        environment.update()

        agent.add_to_replay_mem(five_tup)
        if five_tup[4] is True or five_tup[0] == five_tup[2]:
            still_playing = False
        if MODE != "play":
            if episode % 16 == 0 and episode != 0:
                agent.train_model()
            if episode % 25 == 0:
                agent.target_model.set_weights(agent.model.get_weights())
        recent_state = agent.clean_state_data(five_tup[2])
        agent.episode_num += 1
    agent.epsilon -= agent.epsilon_decay
print(environment.max_score, "MAX SCORE")
agent.save_model()
