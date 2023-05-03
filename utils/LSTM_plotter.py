import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import itertools

log_prefix = "./log/PD_LSTM/run_PD_LSTM_0.1_8_16_1_20230502_200312"
with open(log_prefix + ".log", "r") as f:
    data = f.read()

tag = "recent_avg_loss = "
pos = data.find(tag)
loss_list = []
while pos >= 0:
    data = data[pos+len(tag):]

    pos_n = data.find("\n")
    loss_list.append(float(data[:pos_n]))
    data = data[pos_n+1:]

    pos = data.find(tag)

print(len(loss_list))
y_0, y_1 = [], []
for i in range(len(loss_list) // 2):
    y_0.append(loss_list[2*i])
    y_1.append(loss_list[2*i+1])
assert len(y_0) == len(y_1)

x = np.arange(len(y_0))
y_0 = np.array(y_0, dtype=np.float32)
y_1 = np.array(y_1, dtype=np.float32)
fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(1,1,1)
ax.plot(x, y_0, label="Player 0")
ax.plot(x, y_1, label="Player 1")
ax.legend()
fig.savefig(log_prefix + "_LSTM-loss.png", dpi=200)

exit()
with open(log_prefix + ".pkl", "rb") as f:
    data = pkl.load(f)
print(data.keys())