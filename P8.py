import matplotlib.pyplot as plt
from sklearn.datasets import  load_breast_cancer
from sklearn.tree import DecisionTreeClassifier,plot_tree


data=load_breast_cancer()
X=data.data
y=data.target

model=DecisionTreeClassifier().fit(X,y)

pred=model.predict([X[0]])
if pred==1:
    print("Benign")
else:
    print("Malignant")

plot_tree(model,filled=True)
plt.title("Decision Tree")
plt.show()
