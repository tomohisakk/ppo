import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

from envs.static import MEDAEnv
#from envs.dynamic import MEDAEnv

from PPO import PPO

################################### Training ###################################
def train():
	print("============================================================================================")

	####### initialize environment hyperparameters ######
	env_name ="test"
	env = MEDAEnv(p=0.9)

	max_ep_len = env.max_step                   # max timesteps in one episode
	max_training_timesteps = int(1e8)   # break training loop if timeteps > max_training_timesteps

	print_freq = max_ep_len * 100        # print avg reward in the interval (in num timesteps)
	log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
	save_model_freq = int(1e5)          # save model frequency (in num timesteps)

	#####################################################

	## Note : print/log frequencies should be > than max_ep_len

	################ PPO hyperparameters ################
	update_timestep = max_ep_len * 10      # update policy every n timesteps
	K_epochs = 64               # update policy for K epochs in one PPO update

	eps_clip = 0.1          # clip parameter for PPO
	gamma = 0.99            # discount factor

	lr_actor = 0.00003       # learning rate for actor network
	lr_critic = 0.0001       # learning rate for critic network
	#####################################################

	print("training environment name : " + env_name)

	# state space dimension
	state_dim = 64
#	print(env.observation_space)

	# action space dimension
	action_dim = 4
#		print(env.action_space.n)

	###################### logging ######################

	#### log files for multiple runs are NOT overwritten
	log_dir = "PPO_logs"
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	log_dir = log_dir + '/' + env_name + '/'
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	#### get number of log files in log directory
	run_num = 0
	current_num_files = next(os.walk(log_dir))[2]
	run_num = len(current_num_files)

	#### create new log file for each run
	log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

	print("current logging run number for " + env_name + " : ", run_num)
	print("logging at : " + log_f_name)
	#####################################################

	################### checkpointing ###################
	run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

	directory = "PPO_preTrained"
	if not os.path.exists(directory):
		os.makedirs(directory)

	directory = directory + '/' + env_name + '/'
	if not os.path.exists(directory):
		os.makedirs(directory)


	checkpoint_path = directory + "PPO_{}_{}.pth".format(env_name, run_num_pretrained)
	print("save checkpoint path : " + checkpoint_path)
	#####################################################


	############# print all hyperparameters #############
	print("--------------------------------------------------------------------------------------------")
	print("max training timesteps : ", max_training_timesteps)
	print("max timesteps per episode : ", max_ep_len)
	print("model saving frequency : " + str(save_model_freq) + " timesteps")
	print("log frequency : " + str(log_freq) + " timesteps")
	print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
	print("--------------------------------------------------------------------------------------------")
	print("state space dimension : ", state_dim)
	print("action space dimension : ", action_dim)
	print("--------------------------------------------------------------------------------------------")

	print("Initializing a discrete action space policy")
	print("--------------------------------------------------------------------------------------------")
	print("PPO update frequency : " + str(update_timestep) + " timesteps")
	print("PPO K epochs : ", K_epochs)
	print("PPO epsilon clip : ", eps_clip)
	print("discount factor (gamma) : ", gamma)
	print("--------------------------------------------------------------------------------------------")
	print("optimizer learning rate actor : ", lr_actor)
	print("optimizer learning rate critic : ", lr_critic)
	#####################################################

	print("============================================================================================")

	################# training procedure ################

	# initialize a PPO agent
	ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

	# track total training time
	start_time = datetime.now().replace(microsecond=0)
	print("Started training at (GMT) : ", start_time)

	print("============================================================================================")

	# logging file
	log_f = open(log_f_name,"w+")
	log_f.write('episode,timestep,reward\n')

	# printing and logging variables
	print_running_reward = 0
	print_running_episodes = 0

	log_running_reward = 0
	log_running_episodes = 0

	time_step = 0
	i_episode = 0

	# training loop
	while time_step <= max_training_timesteps:

		state = env.reset()
		current_ep_reward = 0

		for t in range(1, max_ep_len+1):

			# select action with policy
			action = ppo_agent.select_action(state)
			state, reward, done, _ = env.step(action)

			# saving reward and is_terminals
			ppo_agent.buffer.rewards.append(reward)
			ppo_agent.buffer.is_terminals.append(done)

			time_step +=1
			current_ep_reward += reward

			# break; if the episode is over
			if done:
				break

		print_running_reward += current_ep_reward
		print_running_episodes += 1

		log_running_reward += current_ep_reward
		log_running_episodes += 1

		i_episode += 1

		# log in logging file
		if i_episode % 10 == 0:

			# log average reward till last episode
			log_avg_reward = log_running_reward / log_running_episodes
			log_avg_reward = round(log_avg_reward, 4)

			log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
			log_f.flush()

			log_running_reward = 0
			log_running_episodes = 0

		# update PPO agent
		if i_episode % 10 == 0:
			ppo_agent.update()

		# printing average reward
		if i_episode % 100 == 0:

			# print average reward till last episode
			print_avg_reward = print_running_reward / print_running_episodes
			print_avg_reward = round(print_avg_reward, 2)

			print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

			print_running_reward = 0
			print_running_episodes = 0

		# save model weights
		if i_episode % 1000 == 0:
			print("--------------------------------------------------------------------------------------------")
			print("saving model at : " + checkpoint_path)
			ppo_agent.save(checkpoint_path)
			print("model saved")
			print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
			print("--------------------------------------------------------------------------------------------")


	log_f.close()
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
