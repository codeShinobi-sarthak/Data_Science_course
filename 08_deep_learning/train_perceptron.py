from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

x, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = Perceptron(
    max_iter=1000,    # Maximum number of epochs
    eta0=0.1,         # Learning rate
    random_state=42,  # For reproducibility
    tol=1e-3,         # Stop early if improvement is smaller than this
    shuffle=True      # Shuffle data each epoch
)

# training the model
model.fit(x_train, y_train)

# evaluating the model
accuracy = model.score(x_test, y_test)

print(f"Model accuracy: {accuracy * 100:.2f}%")