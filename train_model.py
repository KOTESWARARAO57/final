import os
import pickle
from sklearn.svm import SVC
from extract_features import extract_emotion_from_video

# Directory containing training videos
data_dir = "data/training_data"

X, y = [], []

# Loop through emotion-labeled folders
for label in os.listdir(data_dir):
    folder = os.path.join(data_dir, label)
    if not os.path.isdir(folder):
        continue  # Skip non-folder files
    for file in os.listdir(folder):
        video_path = os.path.join(folder, file)
        emotion = extract_emotion_from_video(video_path)
        X.append([hash(emotion) % 100])  # Simple numeric encoding
        y.append(label)

# Train the classifier
clf = SVC()
clf.fit(X, y)

# ✅ Ensure 'models' directory exists
os.makedirs("models", exist_ok=True)

# ✅ Save the trained model
model_path = "models/model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(clf, f)

print(f"✅ Model trained and saved at {model_path}")
