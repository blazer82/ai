from agent import Agent
from environment import Environment
from model import Model

model = Model(batch_size=32, load=None)
env = Environment(env_type=Environment.TYPE_PONG)
agent = Agent(env=env, model=model)

while True:
	agent.learn(overfit=False)
