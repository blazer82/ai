from agent import Agent
from environment import Environment
from model import Model
import matplotlib.pyplot as plt

model = Model(batch_size=32, lr=1e-4, load=None)
env = Environment(env_type=Environment.TYPE_PONG, render=True)
agent = Agent(env=env, model=model, epsilon_decay=1e-3, min_epsilon=.1)

episode = 0

scores = []
losses = []
qs = []
eps = []
while True:
	episode += 1
	print "Episode #%d"%(episode)
	warmup = 50 if episode == 1 else 0
	score, loss, mean_q, epsilon = agent.learn(overfit=False, games=10, epochs=1, warmup=warmup)

	model.save('model.h5')

	plt.close()

	scores.append(score)
	losses.append(loss)
	qs.append(mean_q)
	eps.append(epsilon)

	s_score = plt.subplot(411)
	s_loss = plt.subplot(412)
	s_q = plt.subplot(413)
	s_eps = plt.subplot(414)

	s_score.set_title('score')
	s_loss.set_title('loss')
	s_q.set_title('mean q')
	s_eps.set_title('epsilon')

	s_score.plot(range(0, len(scores)), scores, 'b-')
	s_loss.plot(range(0, len(losses)), losses, 'b-')
	s_q.plot(range(0, len(qs)), qs, 'b-')
	s_eps.plot(range(0, len(eps)), eps, 'b-')

	plt.show(block=False)
	plt.savefig('plot.png')
