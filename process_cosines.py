import json
import numpy as np
import matplotlib.pyplot as plt

with open('multitask_checkpoints/two/multitask_checkpoints/para_scitail_42_0.0001_cosines.txt', 'r') as f:
    data = json.load(f)
    seed_1 = np.array(data["total"])
    #params = data["parameters"]
    

with open('multitask_checkpoints/two/multitask_checkpoints/para_scitail_13_0.0001_cosines.txt', 'r') as f:    
    data = json.load(f)
    seed_13 = np.array(data["total"])


stance = []
para = []

x = np.arange(len(seed_1))
x2 = np.arange(len(seed_13))
plt.plot(x, seed_1, label='seed = 1')
plt.plot(x2, seed_13, label='seed = 13')
#plt.fill_between(x, np.array(mns) + 3*np.array(stds), np.array(mns) - 3*np.array(stds), alpha=0.5)
plt.xlabel("step")
plt.legend()
plt.ylabel("cosine_similarity")
plt.tight_layout()
#plt.savefig("figs/frac_neg.png")
plt.show()
