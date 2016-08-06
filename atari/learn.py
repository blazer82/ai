from agent import Agent
from environment import Environment
from model import Model

model = Model(batch_size=1024, lr=1e-4, load=None)
env = Environment(env_type=Environment.TYPE_PONG)
agent = Agent(env=env, model=model)

episode = 0
while True:
	episode += 1
	reward = agent.learn(overfit=False)

	print "#%d reward: %d"%(episode, reward)
