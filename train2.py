import os
import glob
import time
from datetime import datetime
import ptan
import torch
import numpy as np

from envs.static import MEDAEnv
#from envs.dynamic import MEDAEnv

from PPO import PPO

from tensorboardX import SummaryWriter

################################### Training ###################################
def train():

	####### initialize environment hyperparameters ######
	env = MEDAEnv(p=0.9)

	K_epochs = 16               # update policy for K epochs in one PPO update

	eps_clip = 0.1          # clip parameter for PPO
	gamma = 0.99            # discount factor

	lr_actor = 1e-9      # learning rate for actor network
	lr_critic = 1e-8       # learning rate for critic network

	env_name = "lr_actor=" + str(lr_actor) + "_critic=" + str(lr_critic)
	writer = SummaryWriter(comment=env_name)
	#####################################################

	print("training environment name : " + env_name)

	state_dim = env.observation_space
#	print(env.observation_space)

	# action space dimension
	action_dim = env.action_space
#		print(env.action_space.n)

	run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

	directory = "PPO_preTrained"
	if not os.path.exists(directory):
		os.makedirs(directory)

	directory = directory + '/' + env_name + '/'
	if not os.path.exists(directory):
		os.makedirs(directory)

	checkpoint_path = directory + "PPO_{}_{}.pth".format(env_name, run_num_pretrained)
	print("save checkpoint path : " + checkpoint_path)

	ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, writer)

	# track total training time
	start_time = datetime.now().replace(microsecond=0)
	print("Started training at (GMT) : ", start_time)

	print("============================================================================================")

	# printing and logging variables
	print_running_reward = 0
	print_running_step = 0
	print_running_episodes = 0

	log_running_reward = 0
	log_running_episodes = 0

	time_step = 0
	i_episode = 0
	i_games = 0
	
	# training loop
	while True:
		while True:
			with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:

				state = env.reset()
				current_ep_reward = 0
				current_ep_step = 0
				done = False
				while done == False:

					# select action with policy
					action = ppo_agent.select_action(state)
					state, reward, done, _ = env.step(action)

					# saving reward and is_terminals
					ppo_agent.buffer.rewards.append(reward)
					ppo_agent.buffer.is_terminals.append(done)

					current_ep_step += 1
					current_ep_reward += reward

				tb_tracker.track("reward", current_ep_reward, i_games)

				print_running_reward += current_ep_reward
				print_running_step += current_ep_step

				print_running_episodes += 1

				log_running_reward += current_ep_reward
				log_running_episodes += 1

				ppo_agent.update(i_games)

				i_games += 1

				# printing average reward
				if i_games % 100 == 0:

					# print average reward till last episode
					print_avg_reward = print_running_reward / print_running_episodes
					print_avg_step = print_running_step / print_running_episodes
					print_avg_reward = round(print_avg_reward, 2)
					print_avg_step = round(print_avg_step, 1)

					print("Episode/Games : {}/{} \t\t Average Steps : {} \t\t Average Reward : {}".format(i_episode, i_games, print_avg_step, print_avg_reward))

					print_running_reward = 0
					print_running_step = 0
					print_running_episodes = 0

				# save model weights
				if i_games % 1000 == 0:
					print("--------------------------------------------------------------------------------------------")
					print("saving model at : " + checkpoint_path)
					ppo_agent.save(checkpoint_path)
					print("model saved")
					print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
					print("--------------------------------------------------------------------------------------------")

				writer.add_scalar("Rewards", current_ep_reward, i_games)
				writer.add_scalar("Steps", current_ep_step, i_games)

				if i_games % 10000 == 0:
					break

		if i_episode == 100:
			break

		i_episode += 1


#	log_f.close()
	env.close()

	# print total training time
	print("============================================================================================")
	end_time = datetime.now().replace(microsecond=0)
	print("Started training at (GMT) : ", start_time)
	print("Finished training at (GMT) : ", end_time)
	print("Total training time  : ", end_time - start_time)
	print("============================================================================================")


if __name__ == '__main__':

	train()
