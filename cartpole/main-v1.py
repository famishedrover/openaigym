# import gym
# env = gym.make('CartPole-v1')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break


# self.state_size 
# self.action_size 
# self.memory
# self.gamma 
# self.epsilon
# self.epsilon_min
# self.epsilon_decay 
# self.learning_rate 
# self.model 

from collections import deque
import numpy as np
import random

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

class DQNAgent :
	def __init__ (self , state_size , action_size) :
		self.state_size = state_size
		self.action_size =action_size
		self.memory = deque(maxlen=2000)
		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.model = self._build_model()

	def _build_model(self):
		model = Sequential()
		model.add(Dense(24, input_dim=self.state_size, activation='relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
		return model
	
	def remember(self ,state ,action ,reward ,next_state ,done):
		self.memory.append((state,action,reward,next_state,done))

	def act(self,state):
		if np.random.rand() <= self.epsilon :
			return random.randrange(self.action_size)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])

	def replay(self,batch_size):
		minibatch = random.sample(self.memory, batch_size)
		for state , action , reward , next_state , done in minibatch :
    		# if done 
			target = reward
			if not done :
				target = reward + self.gamma* np.amax(self.model.predict(next_state)[0])

			target_f = self.model.predict(state)
			target_f[0][action]= target
			self.model.fit(state , target_f , nb_epoch=1 , verbose = 0)
			if self.epsilon > self.epsilon_min :
				self.epsilon *= self.epsilon_decay

game_verison = '1'


if __name__ == "__main__" :
	import gym 


	env = gym.make('CartPole-v'+game_verison)

	agent = DQNAgent(env.observation_space.shape[0] , env.action_space.n)

	episodes = 20000 
	game_frames = 500
	replay_batch_size = 32
	for e in range(episodes) :
		state = env.reset()
		state = np.reshape(state , [1,4])

		for time_t in range(game_frames) :
			# env.render()

			# decide action 
			action = agent.act(state)
			# take action
			next_state , reward , done , _ = env.step(action)
			next_state = np.reshape(next_state , [1,4])

			# remember this action 
			agent.remember(state , action , reward , next_state , done)

			# make next_state as current_state for next frame
			state = next_state

			if done:
				print "episode : {}/{} , score or frame_end : {}".format(e,episodes,time_t)
				break
		if e < 10 :
			replay_batch_size = 5
		else :
			replay_batch_size = 64
		agent.replay(replay_batch_size)















