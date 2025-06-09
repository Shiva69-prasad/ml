import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
def simple_linear_regression():
    data = fetch_california_housing(as_frame=True)
    X = data.data[["AveRooms"]]
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.scatter(X_test, y_test, color="blue", label="Actual")
    plt.plot(X_test, y_pred, color="red", label="Predicted")
    plt.xlabel("AveRooms")
    plt.ylabel("Price ($100k)")
    plt.legend()
    plt.show()
simple_linear_regression()
