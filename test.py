import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

from envs.static import MEDAEnv

from PPO import PPO

def _gen_random_map():
	map = np.random.choice([".", "#", '*'], (8, 8), p=[0.8, 0.1, 0.1])
	map[0][0] = "D"
	map[-1][-1] = "G"

	print("--- Start map ---")
#	print(map)

	return map


#################################### Testing ###################################
def test():
	print("============================================================================================")

	env_name = "static_0915"
	max_ep_len = 100           # max timesteps in one episode

	render = True              # render environment on screen
	frame_delay = 0             # if required; add delay b/w frames

	total_test_episodes = 10    # total num of testing episodes

	K_epochs = 64               # update policy for K epochs
	eps_clip = 0.1              # clip parameter for PPO
	gamma = 0.99                # discount factor

	lr_actor = 0.0001           # learning rate for actor
	lr_critic = 0.001           # learning rate for critic

	#####################################################

	env = MEDAEnv()

	# state space dimension
	state_dim = 64

	# action space dimension
	action_dim = 4

	# initialize a PPO agent
	ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

	# preTrained weights directory

	run_num_pretrained = 0      #### set this to load a particular checkpoint num

	directory = "PPO_preTrained" + '/' + env_name + '/'
	checkpoint_path = directory + "PPO_{}_{}.pth".format(env_name, run_num_pretrained)
	print("loading network from : " + checkpoint_path)

	ppo_agent.load(checkpoint_path)

	print("--------------------------------------------------------------------------------------------")

	test_running_reward = 0

	for ep in range(1, total_test_episodes+1):
		ep_reward = 0
		map = _gen_random_map()
		state = env.reset()

		for t in range(1, max_ep_len+1):
			action = ppo_agent.select_action(state)
			state, reward, done, _ = env.step(action)
			ep_reward += reward

			if done:
				break

		# clear buffer
		ppo_agent.buffer.clear()

		test_running_reward +=  ep_reward
		print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
		ep_reward = 0

	env.close()

	print("============================================================================================")

	avg_test_reward = test_running_reward / total_test_episodes
	avg_test_reward = round(avg_test_reward, 2)
	print("average test reward : " + str(avg_test_reward))

	print("============================================================================================")


if __name__ == '__main__':

	test()
