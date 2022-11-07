import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np

reward = pickle.load(open("reward_1-Ep.pkl","rb"))

print(reward)

new=list(zip(*reward))

x=list(new[0])
y=list(new[1])

print(y)

ysmoothed = gaussian_filter1d(y, sigma=2)

plt.plot(x, ysmoothed)

plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.savefig("reward_Ep.png")




plt.clf()

loss = pickle.load(open("loss_1-Ep.pkl","rb"))

print(loss)

new=list(zip(*loss))

x=list(new[0])
y=list(new[1])

print(y)

ysmoothed = gaussian_filter1d(y, sigma=100)

plt.ylim([0.000, 0.02])
plt.plot(ysmoothed)

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.savefig("loss-Ep.png")

