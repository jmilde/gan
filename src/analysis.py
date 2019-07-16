import tensorflow as tf
import os
d = {}
for f in ["0", "6","9" ]:
    collect =  []
    path=os.path.expanduser("~/Documents/uni/generative models/cache/ae/"+f"dae{f}_b700_btlnk32_lr_0.0001m20")
    x = os.listdir(path)
    for e in tf.compat.v1.train.summary_iterator(path+"/"+x[0]):
        for v in e.summary.value:
            if v.tag == 'AUC_dgx':
                collect.append(v.simple_value)
    d[f]=collect

#35*40 = 1400 datapoints -> 250steps between each
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

fig=plt.figure()
ax=fig.add_subplot(111)
for name, data in d.items():
    ax.plot(np.arange(len(data)),[1-x if x<0.5 else x for x in data], label=name)
ax.set_xlabel("epoch")
ax.set_ylabel("AUC Score")
ax.legend(loc="upper right",ncol=2)
plt.show()
