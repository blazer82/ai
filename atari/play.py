from agent import Agent
from environment import Environment
from model import Model
import matplotlib.pyplot as plt

model = Model(batch_size=1024, lr=1e-4, load='model.h5')
env = Environment(env_type=Environment.TYPE_PONG, render=True)
agent = Agent(env=env, model=model)

episode = 0

scores = []
losses = []
qs = []
eps = []
while True:
	episode += 1
	score = agent.play()

	print "#%d score: %d"%(episode, score)
