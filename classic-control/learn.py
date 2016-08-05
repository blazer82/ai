from agent import Agent
from environment import Environment
from model import Model

model = Model(batch_size=1, lr=1e-1, load=None)
env = Environment(env_type=Environment.TYPE_CART_POLE, render=False)
agent = Agent(env=env, model=model)

episode = 0
while True:
	episode += 1
	reward = agent.learn(overfit=False)

	if reward > 200:
		print "SOLVED after %d episodes!"%(episode)
		while reward > 150:
			reward = agent.play()
			print reward
