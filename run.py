from agent import Agent
from puzzle import GameGrid
import sys
import numpy as np

# normalize input values

episodes = 61234


if len(sys.argv) == 2:
    MODE = sys.argv[1]
else:
    MODE = 'train'

environment = GameGrid()
bot = Agent(MODE, episodes)


# five tup is (state, action, state_after, reward, terminal)
for episode in range(episodes):
    if MODE != "play":
        if episode % 75 == 0 and episode != 0:
            bot.target_model.set_weights(bot.model.get_weights())

    still_playing = True
    state_before_action = environment.give_recent_state()
    step = 0
    while still_playing:

        action = bot.decide_move(state_before_action)
        state_after_action, reward, done = environment.take_action(event=None, action=action)
        bot.remember(np.array((state_before_action, action, state_after_action, reward, done)))
        if done is True:
            still_playing = False

        if MODE != "play":
            bot.train_model()
            print(f"Score: {environment.final_score_prev}; Ep: {episode}; Rand: {round(bot.epsilon, 4)} ")

        state_before_action = state_after_action

    bot.episode_num += 1
    bot.update_epsilon()

print(environment.max_score, "MAX SCORE")
