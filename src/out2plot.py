"Convert slurm-*.out to a plot or something."

from __future__ import unicode_literals, print_function

import sys
from matplotlib import pyplot as plt

plot_fn, = sys.argv[1:]

epoch = None
i_train, loss_train = [], []
i_val, loss_val = [], []
base_i = 0
lines = iter(sys.stdin.readlines())
for line in lines:
    if line.startswith("Epoch "):
        epoch = int(line[6:].split('/')[0])
        statuses = next(lines).split('\r')
        for status in statuses:
            status = status.strip().replace('\x08', '')
            if '/' in status:
                idx_in_epoch = int(status.split('/')[0])
                loss = float(status.split(' - loss: ', 1)[1].split(' - ', 1)[0])
                i_train.append(base_i + idx_in_epoch)
                loss_train.append(loss)
        loss = float(statuses[-1].split(' - val_loss: ', 1)[1])
        i_val.append(i_train[-1])
        loss_val.append(loss)
        base_i = i_train[-1]

plt.plot(i_train, loss_train, 'g-o', label='Training loss')
plt.plot(i_val, loss_val, 'b-o', label='Validation loss')
plt.legend()
plt.xlabel('Training iteration')
plt.ylabel('Categorical cross-entropy loss')
plt.tight_layout()
plt.savefig(plot_fn)
