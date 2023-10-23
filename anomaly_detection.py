import numpy as np
from sklearn.ensemble import IsolationForest

def anomaly_detection(data):
    # Perform anomaly detection using Isolation Forest algorithm
    clf = IsolationForest(contamination=0.05, random_state=42)
    clf.fit(data)

    # Predict anomalies in the data
    anomalies = clf.predict(data)

    # Return indices of anomalous data points
    anomalous_indices = np.where(anomalies == -1)[0]

    return anomalous_indices

# Example usage
data = np.array([[1.2, 3.4, 5.6],
                 [2.3, 4.5, 6.7],
                 [10.1, 12.3, 14.5],
                 [100.2, 200.4, 300.6]])

anomalous_indices = anomaly_detection(data)
print("Anomalous indices:", anomalous_indices)
