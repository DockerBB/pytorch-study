import torch
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = torch.randn((n_train + n_test,1))
poly_features = torch.cat((features,torch.pow(features,2),torch.pow(features,3)),1)
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
          + true_w[2] + poly_features[:, 2] + true_b)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

print(features[:2], poly_features[:2], labels[:2])

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals,y2_vals, linestyle=':')
        d2l.plt.legend(legend)
    plt.show()

num_epochs, loss = 100, torch.nn.MSELoss()

def fit_and_plot(trian_features, test_features, train_labels, test_labels):
    net = torch.nn.Linear(train_labels.shape[-1],1)


