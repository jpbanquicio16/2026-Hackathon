import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

iris = pd.read_csv("iris.csv")

X = iris.drop(columns="species")  # Flower measurements
y = iris["species"]                # Correct species

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Keep the test examples' measurements for your heatmap.
result = X_test.copy()
result.insert(0, "example_id", X_test.index)
result["actual"] = y_test.to_numpy()
result["predicted"] = predictions

result.to_csv("iris_predictions.csv", index=False)

print(f"Exported {len(result)} held-out predictions")
print(f"Accuracy: {(result['actual'] == result['predicted']).mean():.1%}")