from sklearn.model_selection import ParameterGrid
from estimators import pearson_correlation
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from itertools import combinations
import seaborn as sns
import pandas as pd
import tensorflow as tf
from transform import HypersphericalRotation
import numpy as np

data_dict = load_iris(as_frame=True)
y = [data_dict['target_names'][i] for i in data_dict['target'].to_numpy()]
df1 = data_dict['data']
df1['species'] = y

def loss(X):
    result = 0
    for size in range(2, X.shape[1]+1):
        for comb in combinations(range(X.shape[1]), size):
            X_sub = tf.transpose(tf.stack([X[:, i] for i in comb]))
            result += pearson_correlation(X_sub)**2
    return result

x = data_dict['data'].to_numpy()[:,:-1]
x = tf.constant(x, dtype=tf.float64)
model = HypersphericalRotation()
model.set_angles(x)

##tol = 1e-6
##best_loss = np.inf
##best_params = {}
##param_grid = {i:np.linspace(0, 2*np.pi,num=10) for i in range(x.shape[1]-1)}
##
##for grid in ParameterGrid(param_grid):
##    point = [i for i in grid.values()]
##    model.tau = tf.constant(point, dtype=tf.float64)
##    if loss(model(x).numpy()) < best_loss:
##        best_loss = loss(model(x).numpy())
##        best_params = point
##        print(point, loss(model(x).numpy()))
##model.tau = tf.constant(best_params, dtype=tf.float64)
##
##df2 = pd.DataFrame(model(x).numpy())
##df2['species'] = y
##df2 = df2.rename(columns={i:f'Rot({j})' for i,j in zip(range(5), df1.columns)})
##
##sns.pairplot(df1, hue='species')
##sns.pairplot(df2, hue='species')
##plt.show()
    
    
##    plt.plot(history.history['loss'])
##    plt.show()
##    plt.close()
##    for t in weights.T:
##        plt.plot(t)
##        plt.show()
##        plt.close()

history = []
n = 200
for t1 in np.linspace(0, 2 * np.pi, num=n):
    print(t1 / (2 * np.pi))
    for t2 in np.linspace(0, 2 * np.pi, num=n):
        model.tau = tf.constant([t1, t2] + [0] * int(x.shape[1]-3))
        history.append([t1, t2, loss(model(x))])

import scipy.interpolate as interp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

history = np.array(history)
X,Y,Z = history.T

plotx, ploty, = np.meshgrid(np.linspace(np.min(X),np.max(X),n),\
                       np.linspace(np.min(Y),np.max(Y),n))
plotz = interp.griddata((X,Y),Z,(plotx,ploty), method='cubic')

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')

surface_plot = ax1.plot_surface(plotx, ploty, plotz, cmap='coolwarm')
ax1.set_xlabel('$\\tau_1$', fontsize=20)
ax1.set_ylabel('$\\tau_2$', fontsize=20)
ax1.set_zlabel('$\\mathcal{L}$', fontsize=20)

ax2 = fig.add_subplot(122)
contour_plot = ax2.contourf(plotx, ploty, plotz, levels=20, cmap='coolwarm')
ax2.set_xlabel('$\\tau_1$', fontsize=20)
ax2.set_ylabel('$\\tau_2$', fontsize=20)

fig.colorbar(surface_plot, ax=ax1)
plt.tight_layout()
plt.show()
