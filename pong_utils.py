import numpy as np

def preprocessing_observation_of_pong(observation):
    observation_game_court = observation[35:195,:,:]

    down_sample_game_court = observation_game_court[::2, ::2, 0]

    # remove background color
    down_sample_game_court[down_sample_game_court == 144] = 0

    # remove background color
    down_sample_game_court[down_sample_game_court == 109] = 0

    # normalize pong and agent pixels to 1 ?
    down_sample_game_court[down_sample_game_court != 0] = 1

    return down_sample_game_court.astype(np.float).ravel()


def discounted_rewards(reward_list, gamma):
    discounted_rewards = np.zeros_like(reward_list)
    running_add = 0

    for t in reversed(xrange(0, len(reward_list))):
        if reward_list[t] != 0:
            running_add = 0
        running_add = running_add * gamma + reward_list[t]
        discounted_rewards[t] = running_add

    return discounted_rewards