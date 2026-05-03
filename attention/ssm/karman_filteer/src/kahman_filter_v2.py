import numpy as np
import matplotlib.pyplot as plt

# Kalman filter implementation
class KalmanFilter:
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = F  # State transition matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Observation noise covariance
        self.x = x0 # State estimate
        self.P = P0 # Error covariance

    def predict(self):
        # Prediction step
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # Update step
        y = z - self.H @ self.x  # Innovation (difference between observation and prediction)
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

# Parameter settings
dt = 1.0  # Sampling interval
F = np.array([[1, dt],
              [0, 1]])  # State transition matrix: constant velocity motion
H = np.array([[1, 0]])  # Observation matrix: only position is observed
Q = np.array([[0.1, 0],
              [0, 0.1]])  # Process noise (small)
R = np.array([[1.0]])     # Observation noise (large)

# Initial state
x0 = np.array([0, 1.0])  # [position, velocity]
P0 = np.eye(2) * 1.0    # Initial error covariance

# Kalman filter instance
kf = KalmanFilter(F, H, Q, R, x0, P0)

# Data generation
T = 50
true_positions = np.zeros(T)
observed_positions = np.zeros(T)
estimated_positions = np.zeros(T)

# True position: constant velocity motion
for t in range(T):
    true_positions[t] = t * 1.0  # velocity 1.0

# Observation: true position + noise
np.random.seed(42)
observed_positions = true_positions + np.random.normal(0, 1.0, T)

# Kalman filter execution
for t in range(T):
    # Prediction step
    kf.predict()
    # Update step (correct using observation)
    kf.update(observed_positions[t:t+1])
    # Save estimated position
    estimated_positions[t] = kf.x[0]

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(true_positions, label='True position', linestyle='--', color='black')
plt.plot(observed_positions, label='Observed value (with noise)', marker='o', markersize=3, alpha=0.7)
plt.plot(estimated_positions, label='Kalman filter estimate', linewidth=2)
plt.title('Position Estimation with Kalman Filter')
plt.xlabel('Time step')
plt.ylabel('Position')
plt.legend()
plt.grid(True)
plt.show()