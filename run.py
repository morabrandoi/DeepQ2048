from agent import Agent
from puzzle import GameGrid
import sys
import time

# add action replay training
# investigate y_train type
# make randomness decrease
# change verbosity


if len(sys.argv) == 2:
    MODE = sys.argv[1]
else:
    MODE = 'play'

environment = GameGrid()
agent = Agent(MODE)

episodes = 100
# five tup is (state, action, state_after, reward, terminal)
for episode in range(episodes):
    still_playing = True
    recent_state = agent.clean_state_data(environment.give_recent_state())

    while still_playing:
        if episode >= episodes - 2:
            time.sleep(0.3)
        action = agent.decide_move(recent_state)

        five_tup = environment.take_action(event=None, action=action)

        environment.update_idletasks()
        environment.update()

        if five_tup[4] is True or five_tup[0] == five_tup[2]:
            still_playing = False
        if MODE != "play":
            agent.train_model(five_tup)

            if episode % 25 == 0:
                agent.target_model.set_weights(agent.model.get_weights())
        recent_state = agent.clean_state_data(five_tup[2])
