import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.0, random_state=42)
outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
X = np.r_[X, outliers]

clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X)

xx, yy = np.meshgrid(np.linspace(-7, 7, 50), np.linspace(-7, 7, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.title("Detekcja Anomalii Punktowych")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
plt.scatter(X[:, 0], X[:, 1], c='white', s=20, edgecolor='k')
anomalies = X[clf.predict(X) == -1]
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='red', s=50, label='Anomalie')
plt.legend()
plt.show()
