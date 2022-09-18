import torch as T
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = T.device('cpu')
"""
if(T.cuda.is_available()): 
	device = T.device('cuda:0') 
	T.cuda.empty_cache()
	print("Device set to : " + str(T.cuda.get_device_name(device)))
else:
	print("Device set to : cpu")
"""

################################## PPO Policy ##################################
class RolloutBuffer:
	def __init__(self):
		self.actions = []
		self.states = []
		self.logprobs = []
		self.rewards = []
		self.is_terminals = []
	
	def clear(self):
		del self.actions[:]
		del self.states[:]
		del self.logprobs[:]
		del self.rewards[:]
		del self.is_terminals[:]


class ActorCritic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(ActorCritic, self).__init__()

		# actor
		self.actor = nn.Sequential(
						nn.Linear(state_dim, 512),
						nn.Tanh(),
						nn.Linear(512, 128),
						nn.Tanh(),
						nn.Linear(128, action_dim),
						nn.Softmax(dim=-1)
					)
		# critic
		self.critic = nn.Sequential(
						nn.Linear(state_dim, 512),
						nn.Tanh(),
						nn.Linear(512, 128),
						nn.Tanh(),
						nn.Linear(128, 1)
					)

	def forward(self):
		raise NotImplementedError
	
	def act(self, state):
		action_probs = self.actor(state)
		dist = Categorical(action_probs)
		action = dist.sample()

		action_logprob = dist.log_prob(action)

		return action.detach(), action_logprob.detach()
	
	def evaluate(self, state, action):
		action_probs = self.actor(state)
		dist = Categorical(action_probs)
		action_logprobs = dist.log_prob(action)
		dist_entropy = dist.entropy()
		state_values = self.critic(state)
		
		return action_logprobs, state_values, dist_entropy


class PPO:
	def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, writer):
		self.gamma = gamma
		self.eps_clip = eps_clip
		self.K_epochs = K_epochs
		
		self.buffer = RolloutBuffer()

		self.writer = writer

		self.policy = ActorCritic(state_dim, action_dim).to(device)
		self.optimizer = T.optim.Adam([
						{'params': self.policy.actor.parameters(), 'lr': lr_actor},
						{'params': self.policy.critic.parameters(), 'lr': lr_critic}
					])

		self.policy_old = ActorCritic(state_dim, action_dim).to(device)
		self.policy_old.load_state_dict(self.policy.state_dict())

		self.MseLoss = nn.MSELoss()

	def select_action(self, state):
		with T.no_grad():
			state = T.FloatTensor(state).to(device)
			action, action_logprob = self.policy_old.act(state)
		
		self.buffer.states.append(state)
		self.buffer.actions.append(action)
		self.buffer.logprobs.append(action_logprob)

		return action.item()

	def update(self, i_episode):
		# Monte Carlo estimate of returns
		rewards = []
		discounted_reward = 0
		for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
			if is_terminal:
				discounted_reward = 0
			discounted_reward = reward + (self.gamma * discounted_reward)
			rewards.insert(0, discounted_reward)
			
		# Normalizing the rewards
		rewards = T.tensor(rewards, dtype=T.float32).to(device)
		rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

		# convert list to tensor
		old_states = T.squeeze(T.stack(self.buffer.states, dim=0)).detach().to(device)
		old_actions = T.squeeze(T.stack(self.buffer.actions, dim=0)).detach().to(device)
		old_logprobs = T.squeeze(T.stack(self.buffer.logprobs, dim=0)).detach().to(device)

		# Optimize policy for K epochs
		for _ in range(self.K_epochs):

			# Evaluating old actions and values
			logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

			# match state_values tensor dimensions with rewards tensor
			state_values = T.squeeze(state_values)
			
			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = T.exp(logprobs - old_logprobs.detach())

			# Finding Surrogate Loss
			advantages = rewards - state_values.detach()
			surr1 = ratios * advantages
			surr2 = T.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

			loss_value = self.MseLoss(state_values, rewards)
			loss_policy = -T.min(surr1, surr2)
			# final loss of clipped objective PPO
			loss = loss_policy + 0.5*loss_value - 0.01*dist_entropy

			# take gradient step
			self.optimizer.zero_grad()
			loss.mean().backward()
			self.optimizer.step()
		
		loss_f = loss.mean().to(T.float32)
		loss_policy_f = loss_policy.mean().to(T.float32)
		dist_entropy_f = dist_entropy.mean().to(T.float32)

		self.writer.add_scalar("loss_total", loss_f, i_episode)
		self.writer.add_scalar("entropy", dist_entropy_f, i_episode)
		self.writer.add_scalar("loss_value", loss_value, i_episode)
		self.writer.add_scalar("loss_policy", loss_policy_f, i_episode)
		
		# Copy new weights into old policy
		self.policy_old.load_state_dict(self.policy.state_dict())

		# clear buffer
		self.buffer.clear()
	
	def save(self, checkpoint_path):
		T.save(self.policy_old.state_dict(), checkpoint_path)

	def load(self, checkpoint_path):
		self.policy_old.load_state_dict(T.load(checkpoint_path, map_location=lambda storage, loc: storage))
		self.policy.load_state_dict(T.load(checkpoint_path, map_location=lambda storage, loc: storage))

