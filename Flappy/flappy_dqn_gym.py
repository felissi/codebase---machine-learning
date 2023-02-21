# import tensorflow as tf
import numpy as np
import sys
import os
import random
import time
import imageio
import cv2

sys.path.append("game/")
from game import wrapped_flappy_bird as game

sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))

# from layers import *

from PIL import Image
# from scipy.misc import imresize


class flappy():

	def __init__(self):

		# Defining the hyper parameters

		self.img_width = 80
		self.img_height = 80
		self.img_depth = 4
		self.eps = 0.1

		self.num_episodes = 10000
		self.pre_train_steps = 10000
		self.update_freq = 100
		self.batch_size = 32
		self.gamma = 0.99
		self.lr = 0.000001
		self.max_steps = 10000

	def copy_network(self, net1, net2, sess):

		vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, net1.name)
		vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, net2.name)

		for idx in range(len(vars1)):
			sess.run(vars2[idx].assign(vars1[idx]))

	def model(self):

		# Defining the model for the training

		self.target_reward = tf.placeholder(tf.float32, [None, 1], name="target_reward")
		self.action_list = tf.placeholder(tf.int32, [None, 1], name="action_list")

		observed_reward = tf.reduce_sum(self.main_net.q_values*tf.one_hot(tf.reshape(self.action_list,[-1]),2,dtype=tf.float32),1,keep_dims=True)

		self.loss = tf.reduce_mean(tf.square(observed_reward - self.target_reward))

		optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
		self.loss_opt = optimizer.minimize(self.loss)

		self.model_vars = tf.trainable_variables()
		for var in self.model_vars: print(var.name)

	def pre_process(self, img):

		x_t = cv2.cvtColor(cv2.resize(img, (80, 80)), cv2.COLOR_BGR2GRAY)
		ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)

		return x_t

	def policy(self, sess, algo, img_batch):

		if(algo == "e_greedy"):
			
			temp = random.random()
			
			if temp < self.eps :
				temp_action = random.randint(0,1)
			else :
				temp_q_values = sess.run([self.main_net.q_values], 
					feed_dict={self.main_net.input_state:np.reshape(np.stack(img_batch,axis=2),[-1, 80, 80, 4])})
				temp_action = np.argmax(temp_q_values)

			return temp_action

	def train(self):

		self.main_net = network(self.img_width, self.img_height, name="main_net")
		self.target_net = network(self.img_width, self.img_height, name="target_net")

		# Initialising the Networks
		self.main_net.net()
		self.target_net.net()

		self.model()		
		init = tf.global_variables_initializer()

		with tf.Session() as sess:

			sess.run(init)
			print("Initialized the model")
			self.copy_network(self.main_net, self.target_net, sess)
			
			total_steps = 0
			total_reward_list = []
			hist_buffer = []

			# sys.exit()

			for i in range(self.num_episodes):

				# Adding initial 4 frames to the image buffer array

				game_state = game.GameState()
				img_batch = []
				total_reward = 0.0

				temp_action = random.randint(0,1)
				action = np.zeros([2])
				action[temp_action] = 1
				new_state, reward, done = game_state.frame_step(action)

				total_steps+=1
				
				temp_img = self.pre_process(new_state)
				img_batch = [temp_img]*4

				# sys.exit()

				while(True):

					if (total_steps < 100):
						temp_action = random.randint(0,1)
						# print("Temp action is "+ str(temp_action))
					else :
						temp_action = self.policy(sess, "e_greedy", img_batch)
					
					action = np.zeros([2])
					action[temp_action] = 1
					new_state, reward, done = game_state.frame_step(action)
					
					temp_img = self.pre_process(new_state)

					total_reward += reward

					new_img_batch = img_batch[1:]
					new_img_batch.insert(3,temp_img)

					hist_buffer.append((np.stack(img_batch, axis=2), temp_action, reward, np.stack(new_img_batch,axis=2), done))
					
					if(len(hist_buffer) >= 50000):
						hist_buffer.pop(0)

					# Adding the image to the batch

					img_batch.insert(len(img_batch), temp_img)
					img_batch.pop(0)

					# Breaking the loop if the state is terminated


					if(total_steps > self.pre_train_steps):

						rand_batch = random.sample(hist_buffer, self.batch_size)

						reward_hist = [m[2] for m in rand_batch]
						state_hist = [m[0] for m in rand_batch]
						action_hist = [m[1] for m in rand_batch]
						next_state_hist = [m[3] for m in rand_batch]

						temp_target_q = sess.run(self.target_net.q_values, 
							feed_dict={self.target_net.input_state:np.stack(next_state_hist)})

						temp_target_q = np.amax(temp_target_q,1)
						temp_target_reward = reward_hist + self.gamma*temp_target_q
						temp_target_reward =  np.reshape(temp_target_reward, [self.batch_size, 1])

						_ = sess.run(self.loss_opt, feed_dict={self.main_net.input_state:np.stack(state_hist), 
							self.target_reward:temp_target_reward, 
							self.action_list:np.reshape(np.stack(action_hist),[self.batch_size, 1])})

						if(total_steps%self.update_freq == 0):
							self.copy_network(self.main_net, self.target_net, sess)

					if (done):
						break

					total_steps+=1
				
				print("Total rewards in episode " + str(i) + " is " + str(total_reward) + " total number of steps are " + str(total_steps))
				# sys.exit()
			# for var in self.model_vars: print(var.name, sess.run(var.name))

	def play(self, mode="random"):


		env = game.GameState()
		total_steps = 0
		img_batch = []

		action = np.zeros([2])
		action[1] = 1
		new_state, reward, done =  env.frame_step(action)

		temp_img = self.pre_process(new_state)

		for j in range(4):
			img_batch.insert(len(img_batch), temp_img)
		
		for j in range(self.max_steps):

			temp_action = random.randint(0,1)
				
			action = np.zeros([2])
			action[temp_action] = 1

			new_state, reward, done =  env.frame_step(action)

			temp_new_state = np.flip(np.rot90(new_state, k=1, axes=(1,0)), 1)

			temp_img = self.pre_process(new_state)
			img_batch.insert(0, temp_img)
			img_batch.pop(len(img_batch)-1)

			print(temp_action)

			total_steps += 1
			
			if done:
				break

		print("Total Steps ", str(total_steps))

		sys.exit()

def main():

	mod = flappy()
	# mod.train()
	mod.play("random")

if __name__ == "__main__":
	main()
