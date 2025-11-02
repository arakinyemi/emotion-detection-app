import os
import cv2
import numpy as np
import os
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


train_dir = '/content/fer2013/train'
test_dir = '/content/fer2013/test'

print("Train folders:", os.listdir(train_dir))
print("Test folders:", os.listdir(test_dir))



def load_data(data_dir):
    X, y = [], []
    classes = os.listdir(data_dir)

    for label, emotion in enumerate(classes):
        emotion_dir = os.path.join(data_dir, emotion)
        for img_name in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48))
            X.append(img.flatten())   # flatten for scikit-learn
            y.append(label)
    
    return np.array(X), np.array(y), classes

X_train, y_train, class_names = load_data(train_dir)
X_test, y_test, _ = load_data(test_dir)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))
print("Classes:", class_names)

X_train = X_train / 255.0
X_test = X_test / 255.0

mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    batch_size=256,
    learning_rate_init=0.001,
    max_iter=50,
    verbose=True,
    random_state=42
)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

joblib.dump(mlp, "emotion_mlp_model.pkl")

from google.colab import files
files.download("emotion_mlp_model.pkl")