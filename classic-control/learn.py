from agent import Agent
from environment import Environment
from model import Model

model = Model(batch_size=1, load=None)
env = Environment(env_type=Environment.TYPE_CART_POLE, render=True)
agent = Agent(env=env, model=model)

while True:
	agent.learn(overfit=False)
