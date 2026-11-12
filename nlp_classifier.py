
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

# Load labeled training data
data = pd.read_csv("training_data.csv")

# Define our reasoning categories
categories = [
    "visual_audio_clues",
    "source_verification",
    "independent_corroboration",
    "uncertainty_awareness"
]

# Load pretrained embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert explanations into embeddings
X = embedding_model.encode(data["explanation"].tolist())

# Train one classifier for each category
classifiers = {}

for category in categories:
    y = data[category]

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X, y)

    classifiers[category] = classifier

print("All four classifiers trained successfully!")


# Predict strategies in new explanations
def predict_strategies(explanation):
    embedding = embedding_model.encode([explanation])

    predictions = {}

    for category, classifier in classifiers.items():
        prediction = classifier.predict(embedding)[0]
        predictions[category] = bool(prediction)

    return predictions


# Test our classifier
if __name__ == "__main__":
    example = (
        "The lips don't match the audio, "
        "but I would check the original video."
    )

    results = predict_strategies(example)

    print("\nParticipant explanation:")
    print(example)

    print("\nDetected reasoning strategies:")
    for category, detected in results.items():
        print(f"{category}: {detected}")
