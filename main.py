import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models  # pyright: ignore[reportMissingImports]
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pickle
import json
import shap
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from evaluation import generate_evaluation_report

warnings.filterwarnings("ignore")

np.random.seed(42)
tf.random.set_seed(42)


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
        # numerical features
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


class TransformerBlock(layers.Layer):
    """
    Transformer block with multi-head attention
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

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
    """
    Custom attention mechanism for feature importance
    """

    def __init__(self, units, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = layers.Dense(self.units)
        self.V = layers.Dense(1)
        super().build(input_shape)

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


def build_hybrid_model(input_shape, num_classes=2):
    """
    Build hybrid CNN + Transformer + Attention model

    Architecture:
    1. CNN layers: Extract local patterns from biometric features
    2. Transformer blocks: Capture long-range dependencies
    3. Attention mechanism: Focus on important features
    4. Dense layers: Final classification
    """

    # Input layer
    inputs = layers.Input(shape=input_shape)

    # Reshape for CNN (add channel dimension)
    x = layers.Reshape((input_shape[0], 1))(inputs)

    # ===== CNN BLOCK =====
    # First convolutional block
    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    # Second convolutional block
    x = layers.Conv1D(filters=128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    # ===== TRANSFORMER BLOCK =====
    # Prepare for transformer (needs 3D input: batch, sequence, features)
    seq_length = x.shape[1]
    embed_dim = x.shape[2]

    # Add positional encoding
    positions = tf.range(start=0, limit=seq_length, delta=1)
    position_embedding = layers.Embedding(input_dim=seq_length, output_dim=embed_dim)(
        positions
    )
    x = x + position_embedding

    # Apply transformer blocks
    x = TransformerBlock(embed_dim=embed_dim, num_heads=4, ff_dim=256)(x)
    x = TransformerBlock(embed_dim=embed_dim, num_heads=4, ff_dim=256)(x)

    # ===== ATTENTION MECHANISM =====
    context_vector = AttentionLayer(units=128)(x)

    # ===== DENSE CLASSIFICATION LAYERS =====
    x = layers.Dense(256, activation="relu")(context_vector)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Output layer
    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Create model
    model = models.Model(
        inputs=inputs, outputs=outputs, name="HybridInsiderThreatDetector"
    )

    return model


def find_optimal_threshold(y_true, y_pred_proba, target_recall=0.75):
    """Find threshold that achieves target recall with best precision"""
    from sklearn.metrics import precision_score, recall_score

    thresholds = np.arange(0.3, 0.9, 0.05)
    best_threshold = 0.5
    best_precision = 0

    print("\nThreshold Tuning:")
    print("Threshold | Precision | Recall | F1-Score")
    print("-" * 50)

    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int).flatten()
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print(f"{threshold:.2f}      | {precision:.4f}    | {recall:.4f} | {f1:.4f}")

        # Find best threshold that keeps recall >= target
        if recall >= target_recall and precision > best_precision:
            best_precision = precision
            best_threshold = threshold

    print(f"\nBest threshold: {best_threshold:.2f} (Precision: {best_precision:.4f})")
    return best_threshold


def train_model(df, test_size=0.2, epochs=20, batch_size=256):  # 50 default
    """
    Train the hybrid model with proper validation and metrics
    """
    print("=" * 80)
    print("STARTING MODEL TRAINING")
    print("=" * 80)

    # Preprocess data
    print("\n[1/6] Preprocessing data...")
    preprocessor = BiometricPreprocessor()
    X = preprocessor.fit_transform(df)
    y = df["label"].values

    print(f"Features shape: {X.shape}")
    print(f"Labels distribution: {np.bincount(y)}")

    # Split data
    print("\n[2/6] Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Build model
    print("\n[3/6] Building hybrid model architecture...")
    input_shape = (X_train.shape[1],)
    model = build_hybrid_model(input_shape)

    class_weights = compute_class_weight(  # noqa: F841
        "balanced", classes=np.unique(y_train), y=y_train
    )
    # class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    class_weight_dict = {
        0: 1.0,
        1: 5.0,
    }

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    print("\nModel Architecture:")
    model.summary()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,  # Changed from 10 to 5
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=0.00001,  # Changed from 5 to 3
    )

    # Train model
    print("\n[4/6] Training model...")
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    # Evaluate model
    print("\n[5/6] Evaluating model on test set...")
    test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(
        X_test, y_test, verbose=0
    )

    print("\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  AUC: {test_auc:.4f}")

    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()  # threshold at 0.5
    optimal_threshold = find_optimal_threshold(y_test, y_pred_proba, target_recall=0.75)
    y_pred = (y_pred_proba > optimal_threshold).astype(int).flatten()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Threat"]))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Save model and preprocessor
    print("\n[6/6] Saving model and preprocessor...")
    model.save("insider_threat_model.h5")

    with open("preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    # Save model metadata
    metadata = {
        "feature_names": preprocessor.feature_names,
        "input_shape": input_shape,
        "test_accuracy": float(test_acc),
        "test_auc": float(test_auc),
        "training_date": datetime.now().isoformat(),
        "num_features": X.shape[1],
    }

    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nModel saved successfully!")
    print("  - insider_threat_model.h5")
    print("  - preprocessor.pkl")
    print("  - model_metadata.json")

    return model, preprocessor, history, X_test, y_test, y_pred_proba


def explain_predictions(model, preprocessor, X_test, feature_names, num_samples=100):
    """
    Generate SHAP explanations for model predictions using KernelExplainer
    """
    X_sample = X_test[:num_samples]
    X_background = X_test[100:200]

    print(f"\nAnalyzing {num_samples} samples with SHAP KernelExplainer...")

    def model_predict(x):
        return model.predict(x, verbose=0).flatten()

    explainer = shap.KernelExplainer(model_predict, X_background)
    shap_values = explainer.shap_values(X_sample, nsamples=100)

    print("\nGenerating SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title("Feature Importance for Insider Threat Detection")
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=300, bbox_inches="tight")
    print("Saved: shap_summary.png")

    mean_shap = np.abs(shap_values).mean(axis=0)

    # feature importance dataframe
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": mean_shap}
    ).sort_values("importance", ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))

    feature_importance.to_csv("feature_importance.csv", index=False)
    print("\nSaved: feature_importance.csv")

    return explainer, shap_values, feature_importance


if __name__ == "__main__":
    print("=" * 80)

    print("\nLoading dataset...")
    df = pd.read_csv("./dataset/biometric_train_v2.csv")

    # train
    model, preprocessor, history, X_test, y_test, y_pred = train_model(df)
    
    generate_evaluation_report(X_test, y_test)

    explainer, shap_values, importance = explain_predictions(
        model, preprocessor, X_test, preprocessor.feature_names
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
