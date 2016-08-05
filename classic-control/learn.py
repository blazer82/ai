from agent import Agent
from environment import Environment
from model import Model

model = Model(batch_size=128, lr=1e-2, load=None)
env = Environment(env_type=Environment.TYPE_CART_POLE, render=False)
agent = Agent(env=env, model=model)

episode = 0
first_reward = 0
while True:
	episode += 1
	reward = agent.learn(overfit=False)

	if first_reward == 0:
		first_reward = reward

	print "Reward delta: %d"%(reward - first_reward)

	if reward > 200:
		print "SOLVED after %d episodes!"%(episode)
		while reward > 150:
			reward = agent.play()
			print reward
