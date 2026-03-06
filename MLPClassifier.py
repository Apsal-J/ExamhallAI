import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

def get_features(kps):
    kps = np.array(kps)[:, :2]  # x,y only
    center = np.mean(kps, axis=0)
    dist = np.linalg.norm(kps - center, axis=1)
    return dist  # length 17

X = []
y = []

activities = {"standing":0, "turning around":1, "normal":2}

for activity, cid in activities.items():
    kp_dir = f"finaldataone/{activity}/keypoints"
    for file in os.listdir(kp_dir):
        raw = list(map(float, open(f"{kp_dir}/{file}").read().split()))
        kps = np.array(raw).reshape(-1,2)
        f = get_features(kps)
        X.append(f)
        y.append(cid)

X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', max_iter=500)
clf.fit(X_scaled, y)

joblib.dump(clf, "pose_classifiera.pkl")
joblib.dump(scaler, "scalera.pkl")
