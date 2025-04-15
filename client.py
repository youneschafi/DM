from ucimlrepo import fetch_ucirepo
import requests

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features.copy()
y = heart_disease.data.targets.copy()
y = (y > 0).astype(int).values.ravel()

sample = X.iloc[:5]
true_labels = y[:5]
records = sample.to_dict(orient="records")

url = "http://127.0.0.1:8000/predict"
predictions = []

for record in records:
    response = requests.post(url, json=record)
    predictions.append(response.json()["has_heart_disease"])

correct = sum(pred==true for pred, true in zip(predictions, true_labels))
accuracy = correct / len(true_labels)

rating = "excellent" if accuracy == 1 else "good" if accuracy >= 0.8 else "fair" if accuracy >= 0.5 else "poor"

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Rating: {rating}")
