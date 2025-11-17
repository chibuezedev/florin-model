from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore
import numpy as np
import pandas as pd
import pickle
import json
import shap
from datetime import datetime
import logging
import traceback
from sklearn.preprocessing import StandardScaler, LabelEncoder
tf.keras.backend.clear_session()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

model = None
preprocessor = None
metadata = None
explainer = None

custom_objects = {"TransformerBlock": None, "AttentionLayer": None}


class TransformerBlock(keras.layers.Layer):
    """Transformer block - must match training definition"""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # KEEP THESE IN __init__ - matching training code
        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class AttentionLayer(layers.Layer):
    """Custom attention mechanism for feature importance"""

    def __init__(self, units, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        # KEEP THESE IN __init__ - matching training code
        self.W = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, inputs):
        score = self.V(tf.nn.tanh(self.W(inputs)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


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


def build_hybrid_model(input_shape, num_classes=2):
    """
    Rebuild the model architecture - COPY THIS FROM YOUR TRAINING SCRIPT
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0], 1))(inputs)

    # CNN blocks
    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(filters=128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    # Transformer preparation
    seq_length = x.shape[1]
    embed_dim = x.shape[2]

    positions = tf.range(start=0, limit=seq_length, delta=1)
    position_embedding = layers.Embedding(input_dim=seq_length, output_dim=embed_dim)(
        positions
    )
    x = x + position_embedding

    # Transformer blocks
    x = TransformerBlock(embed_dim=embed_dim, num_heads=4, ff_dim=256)(x)
    x = TransformerBlock(embed_dim=embed_dim, num_heads=4, ff_dim=256)(x)

    # Attention
    context_vector = AttentionLayer(units=128)(x)

    # Dense layers
    x = layers.Dense(256, activation="relu")(context_vector)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.models.Model(
        inputs=inputs, outputs=outputs, name="HybridInsiderThreatDetector"
    )
    return model


def load_model_and_preprocessor():
    """Load trained model and preprocessor on startup"""
    global model, preprocessor, metadata

    try:
        logger.info("Loading model and preprocessor...")

        # Load preprocessor first to get input shape
        with open("preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        logger.info("Preprocessor loaded successfully")

        # Load metadata to get input shape
        with open("model_metadata.json", "r") as f:
            metadata = json.load(f)
        logger.info(f"Model metadata loaded: {metadata}")

        # Rebuild model architecture
        input_shape = tuple(metadata["input_shape"])
        model = build_hybrid_model(input_shape)

        # Compile model (needed before loading weights)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        # Load only the weights
        model.load_weights("insider_threat_model.h5")
        logger.info("Model weights loaded successfully")

        return True

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        return False


load_model_and_preprocessor()

def extract_features(data):
    """
    Extract and format features from incoming request data
    Matches the format expected by your Node.js middleware
    """
    features = {
        # Logon patterns
        "timeOfDay": data.get("logonTimeOfDay", 12),
        "dayOfWeek": data.get("logonDayOfWeek", 3),
        # Typing dynamics
        "wpm": data.get("typingSpeed", 40),
        "typingAccuracy": 95.0,
        "dwellTime_mean": data.get("typingDwellTime", 100),
        "flightTime_mean": data.get("typingFlightTime", 150),
        # Mouse dynamics
        "mouseDynamics_velocity": data.get("mouseVelocity", 500),
        "mouseDynamics_acceleration": data.get("mouseAcceleration", 200),
        "mouseDynamics_curvature": data.get("mouseCurvature", 0.5),
        # Touch gestures (mobile)
        "touchPressure": data.get("touchPressure", 0.5),
        "swipeVelocity": data.get("touchSwipeVelocity", 300),
        # Email patterns
        "locationConsistency": data.get("emailSendTimeConsistency", 80),
        # Categorical features - from Node.js request
        "archetype": data.get("archetype", "casual_user"),
        "activityFlow": data.get("activityFlow", "normal_routine"),
        # Other features
        "loginDuration": 300,
        "failedAttempts": 0,
        "dwellTime_std": 20,
        "flightTime_std": 30,
        "clickFrequency": 5,
        "scrollSpeed": 100,
    }

    return features


def calculate_risk_level(anomaly_score):
    """
    Convert anomaly score to risk level
    """
    if anomaly_score >= 80:
        return "critical"
    elif anomaly_score >= 60:
        return "high"
    elif anomaly_score >= 40:
        return "medium"
    else:
        return "low"


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": model is not None,
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint
    Expects JSON with behavioral biometric features
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        logger.info(
            f"Received prediction request for user: {data.get('userId', 'unknown')}"
        )

        features = extract_features(data)

        # Convert to DataFrame for preprocessing
        df_input = pd.DataFrame([features])

        # Preprocess features
        X = preprocessor.transform(df_input)

        # Make prediction
        prediction_proba = model.predict(X, verbose=0)[0][0]
        anomaly_score = float(prediction_proba * 100)

        # Determine risk level
        risk_level = calculate_risk_level(anomaly_score)

        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = model(X_tensor, training=False)

        gradients = tape.gradient(predictions, X_tensor)
        gradient_values = gradients.numpy()[0]

        feature_importance = []
        for i, (feature, value) in enumerate(
            zip(preprocessor.feature_names, gradient_values)
        ):
            feature_importance.append(
                {
                    "feature": feature,
                    "importance": float(value),
                    "absImportance": float(abs(value)),
                }
            )

        feature_importance.sort(key=lambda x: x["absImportance"], reverse=True)

        response = {
            "anomalyScore": round(float(anomaly_score), 2),
            "riskLevel": risk_level,
            "prediction": "threat" if anomaly_score >= 65 else "normal",
            "confidence": round(float(abs(prediction_proba - 0.5) * 200), 2),
            "userId": data.get("userId", "unknown"),
            "featureImportance": feature_importance[:10],  # Top 10
            "explanation": f"The model detected a {risk_level} risk level. "
            f"Key factors: {', '.join([f['feature'] for f in feature_importance[:3]])}",
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Prediction completed: {response['riskLevel']} (score: {response['anomalyScore']})"
        )

        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Prediction failed", "message": str(e)}), 500


@app.route("/explain", methods=["POST"])
def explain():
    """
    Explain prediction using SHAP values
    Returns feature importance for the prediction
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        logger.info(
            f"Received explanation request for user: {data.get('userId', 'unknown')}"
        )

        # Extract and preprocess features
        features = extract_features(data)
        df_input = pd.DataFrame([features])
        X = preprocessor.transform(df_input)

        # Make prediction
        prediction_proba = model.predict(X, verbose=0)[0][0]
        anomaly_score = float(prediction_proba * 100)
        risk_level = calculate_risk_level(anomaly_score)

        background = np.zeros((10, X.shape[1]))
        local_explainer = shap.DeepExplainer(model, background)
        shap_values = local_explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_vals = shap_values[0][0]
        else:
            shap_vals = shap_values[0]

        feature_importance = []
        for i, (feature, value) in enumerate(
            zip(preprocessor.feature_names, shap_vals)
        ):
            feature_importance.append(
                {
                    "feature": feature,
                    "importance": float(value),
                    "absImportance": float(abs(value)),
                }
            )

        feature_importance.sort(key=lambda x: x["absImportance"], reverse=True)

        response = {
            "anomalyScore": round(anomaly_score, 2),
            "riskLevel": risk_level,
            "prediction": "threat" if anomaly_score >= 65 else "normal",
            "featureImportance": feature_importance[:10],  # Top 10
            "explanation": f"The model detected a {risk_level} risk level. "
            f"Key factors: {', '.join([f['feature'] for f in feature_importance[:3]])}",
            "timestamp": datetime.now().isoformat(),
        }

        logger.info("Explanation generated successfully")

        return jsonify(response)

    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Explanation failed", "message": str(e)}), 500


@app.route("/debug/labels", methods=["GET"])
def debug_labels():
    """Debug endpoint to see valid categorical values"""
    if preprocessor:
        valid_labels = {}
        for col, encoder in preprocessor.label_encoders.items():
            valid_labels[col] = list(encoder.classes_)
        return jsonify(valid_labels)
    return jsonify({"error": "Preprocessor not loaded"}), 503


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    Batch prediction endpoint for multiple samples
    """
    try:
        data = request.get_json()

        if not data or "samples" not in data:
            return jsonify({"error": "No samples provided"}), 400

        samples = data["samples"]
        logger.info(f"Received batch prediction request with {len(samples)} samples")

        predictions = []

        for sample in samples:
            features = extract_features(sample)
            df_input = pd.DataFrame([features])
            X = preprocessor.transform(df_input)

            prediction_proba = model.predict(X, verbose=0)[0][0]
            anomaly_score = float(prediction_proba * 100)

            predictions.append(
                {
                    "userId": sample.get("userId", "unknown"),
                    "anomalyScore": round(anomaly_score, 2),
                    "riskLevel": calculate_risk_level(anomaly_score),
                    "prediction": "threat" if anomaly_score >= 65 else "normal",
                }
            )

        return jsonify(
            {
                "predictions": predictions,
                "count": len(predictions),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({"error": "Batch prediction failed", "message": str(e)}), 500


@app.route("/model_info", methods=["GET"])
def model_info():
    """
    Get model information and metadata
    """
    if metadata:
        return jsonify(
            {
                "model_metadata": metadata,
                "status": "loaded",
                "endpoints": {
                    "predict": "/predict",
                    "explain": "/explain",
                    "batch_predict": "/batch_predict",
                    "health": "/health",
                },
            }
        )
    else:
        return jsonify({"error": "Model not loaded"}), 503


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    success = load_model_and_preprocessor()

    if not success:
        logger.error("Failed to load model. Server will not start.")
        exit(1)

    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
