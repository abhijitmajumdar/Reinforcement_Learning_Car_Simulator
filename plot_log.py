import numpy as np
import matplotlib.pyplot as plt

log = np.reshape(np.load('./log.npy'),(-1))[0]
keys = [k for k in log]
l = len(log[keys[0]])
epochs = range(l)
offset = 0

fig = plt.figure()
ax1 = plt.subplot(211)
ax1.plot(epochs[offset:l],log['final_score'][offset:l],linestyle='--',label='final score',linewidth=2,c='blue')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Final score')
ax1.legend(loc=3)
ax2 = ax1.twinx()
ax2.plot(epochs[offset:l],log['avg_loss'][offset:l],linestyle='-.',label='avg loss',linewidth=2,c='red')
ax2.set_ylabel('Average loss')
ax2.legend(loc=2)

ax3 = plt.subplot(212)
ax3.set_xlabel('Epochs')
ax3.plot(epochs[offset:l],log['cross_score'][offset:l],linestyle='-',label='cross score',linewidth=2,c='green')
ax3.set_ylabel('Cross score')
ax3.legend()

plt.show()
