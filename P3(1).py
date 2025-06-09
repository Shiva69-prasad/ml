from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
iris = load_iris() # make sure use ()
X, y = iris.data, iris.target
X_pca = PCA(n_components=2).fit_transform(X)
for i, c in enumerate('rgb'):
    plt.scatter(*X_pca[y == i].T, c=c, label=iris.target_names[i])
plt.legend()
plt.title('PCA - Iris')
plt.show()
