import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
california_data=fetch_california_housing(as_frame=True)
data=california_data.frame
correlation_matrix=data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True)
plt.title('correlation matrix of california_housing feature')
plt.show()

sns.pairplot(data,diag_kind='kde',plot_kws={'alpha':0.5})
plt.suptitle('pair plot of california_housing_feature',y=1.02)
plt.show()
