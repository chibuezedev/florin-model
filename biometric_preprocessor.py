import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

tf.keras.backend.clear_session()


class BiometricPreprocessor:
    """
    Handles data preprocessing for behavioral biometric features
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []

    def fit_transform(self, df):
        """
        Fit preprocessor and transform data
        """
        numerical_features = [
            "timeOfDay",
            "dayOfWeek",
            "loginDuration",
            "failedAttempts",
            "locationConsistency",
            "wpm",
            "typingAccuracy",
            "mouseDynamics_velocity",
            "mouseDynamics_acceleration",
            "mouseDynamics_curvature",
            "dwellTime_mean",
            "dwellTime_std",
            "flightTime_mean",
            "flightTime_std",
            "clickFrequency",
            "scrollSpeed",
            "touchPressure",
            "swipeVelocity",
        ]

        # categorical features
        categorical_features = ["archetype", "activityFlow"]

        # Process numerical features - only those that exist in the dataframe
        available_numerical = [f for f in numerical_features if f in df.columns]
        X_numerical = df[available_numerical].values
        X_numerical_scaled = self.scaler.fit_transform(X_numerical)

        # Process categorical features
        X_categorical = []
        for col in categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                encoded = le.fit_transform(df[col].astype(str))
                X_categorical.append(encoded.reshape(-1, 1))
                self.label_encoders[col] = le

        X_categorical = (
            np.hstack(X_categorical)
            if X_categorical
            else np.array([]).reshape(len(df), 0)
        )

        # Combine features
        X_combined = np.hstack([X_numerical_scaled, X_categorical])

        # Store feature names
        self.feature_names = available_numerical + categorical_features

        return X_combined

    def transform(self, df):
        """
        Transform new data using fitted preprocessor
        """
        numerical_features = [
            "timeOfDay",
            "dayOfWeek",
            "loginDuration",
            "failedAttempts",
            "locationConsistency",
            "wpm",
            "typingAccuracy",
            "anomalyScore",
            "mouseDynamics_velocity",
            "mouseDynamics_acceleration",
            "mouseDynamics_curvature",
            "dwellTime_mean",
            "dwellTime_std",
            "flightTime_mean",
            "flightTime_std",
            "clickFrequency",
            "scrollSpeed",
            "touchPressure",
            "swipeVelocity",
        ]

        categorical_features = ["archetype", "activityFlow", "riskLevel"]

        available_numerical = [f for f in numerical_features if f in df.columns]
        X_numerical = df[available_numerical].values
        X_numerical_scaled = self.scaler.transform(X_numerical)

        X_categorical = []
        for col in categorical_features:
            if col in df.columns and col in self.label_encoders:
                encoded = self.label_encoders[col].transform(df[col].astype(str))
                X_categorical.append(encoded.reshape(-1, 1))

        X_categorical = (
            np.hstack(X_categorical)
            if X_categorical
            else np.array([]).reshape(len(df), 0)
        )

        return np.hstack([X_numerical_scaled, X_categorical])
