
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
)
from tensorflow import keras
import pickle
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


class ModelEvaluator:
    """
    Comprehensive model evaluation class
    """

    def __init__(
        self, model_path="./model/insider_threat_model.h5", preprocessor_path="./model/preprocessor.pkl"
    ):
        """
        Initialize evaluator with trained model
        """
        print("Loading model and preprocessor...")

        from flask_api import TransformerBlock, AttentionLayer

        custom_objects = {
            "TransformerBlock": TransformerBlock,
            "AttentionLayer": AttentionLayer,
        }

        self.model = keras.models.load_model(model_path, custom_objects=custom_objects)

        with open(preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)

        print("Model and preprocessor loaded successfully!")

    def evaluate_comprehensive(self, X_test, y_test, save_plots=True):
        """
        Comprehensive evaluation with multiple metrics
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 80)

        print("\n[1/7] Generating predictions...")
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Basic metrics
        print("\n[2/7] Calculating performance metrics...")
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

        print("\n=== PERFORMANCE METRICS ===")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Classification report
        print("\n[3/7] Generating classification report...")
        print("\n=== CLASSIFICATION REPORT ===")
        print(
            classification_report(
                y_test, y_pred, target_names=["Normal", "Threat"], digits=4
            )
        )

        # Confusion matrix
        print("\n[4/7] Creating confusion matrix...")
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm, save=save_plots)

        # ROC curve
        print("\n[5/7] Plotting ROC curve...")
        self._plot_roc_curve(y_test, y_pred_proba, save=save_plots)

        # Precision-Recall curve
        print("\n[6/7] Plotting Precision-Recall curve...")
        self._plot_precision_recall_curve(y_test, y_pred_proba, save=save_plots)

        # Threshold analysis
        print("\n[7/7] Analyzing decision thresholds...")
        self._analyze_thresholds(y_test, y_pred_proba, save=save_plots)

        # Save metrics
        self._save_metrics(metrics)

        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)

        return metrics

    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """
        Calculate comprehensive metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            roc_auc_score,
            average_precision_score,
        )

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
            "avg_precision": average_precision_score(y_true, y_pred_proba),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        }

        # Calculate specificity and sensitivity
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0

        return metrics

    def _plot_confusion_matrix(self, cm, save=True):
        """
        Plot confusion matrix heatmap
        """
        plt.figure(figsize=(10, 8))

        # Normalize confusion matrix
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            xticklabels=["Normal", "Threat"],
            yticklabels=["Normal", "Threat"],
            cbar_kws={"label": "Percentage"},
        )

        plt.title("Confusion Matrix (Normalized)", fontsize=16, fontweight="bold")
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)

        # Add counts
        for i in range(2):
            for j in range(2):
                plt.text(
                    j + 0.5,
                    i + 0.7,
                    f"n={cm[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="gray",
                )

        plt.tight_layout()

        if save:
            plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
            print("  Saved: confusion_matrix.png")

        plt.close()

    def _plot_roc_curve(self, y_true, y_pred_proba, save=True):
        """
        Plot ROC curve
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
        )
        plt.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=2,
            linestyle="--",
            label="Random Classifier",
        )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title(
            "Receiver Operating Characteristic (ROC) Curve",
            fontsize=16,
            fontweight="bold",
        )
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save:
            plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
            print("  Saved: roc_curve.png")

        plt.close()

    def _plot_precision_recall_curve(self, y_true, y_pred_proba, save=True):
        """
        Plot Precision-Recall curve
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)

        plt.figure(figsize=(10, 8))
        plt.plot(
            recall,
            precision,
            color="blue",
            lw=2,
            label=f"PR curve (AP = {avg_precision:.4f})",
        )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("Precision-Recall Curve", fontsize=16, fontweight="bold")
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save:
            plt.savefig("precision_recall_curve.png", dpi=300, bbox_inches="tight")
            print("  Saved: precision_recall_curve.png")

        plt.close()

    def _analyze_thresholds(self, y_true, y_pred_proba, save=True):
        """
        Analyze different decision thresholds
        """
        thresholds = np.arange(0.1, 1.0, 0.05)

        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int).flatten()

            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
            )

            accuracies.append(accuracy_score(y_true, y_pred))
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))

        plt.figure(figsize=(12, 8))
        plt.plot(thresholds, accuracies, label="Accuracy", marker="o", linewidth=2)
        plt.plot(thresholds, precisions, label="Precision", marker="s", linewidth=2)
        plt.plot(thresholds, recalls, label="Recall", marker="^", linewidth=2)
        plt.plot(thresholds, f1_scores, label="F1 Score", marker="d", linewidth=2)

        plt.axvline(
            x=0.5, color="red", linestyle="--", alpha=0.5, label="Default (0.5)"
        )

        plt.xlabel("Decision Threshold", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.title("Metrics vs Decision Threshold", fontsize=16, fontweight="bold")
        plt.legend(loc="best", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save:
            plt.savefig("threshold_analysis.png", dpi=300, bbox_inches="tight")
            print("  Saved: threshold_analysis.png")

        plt.close()

        # Find optimal threshold (maximize F1)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]

        print(f"\n  Optimal threshold (max F1): {optimal_threshold:.2f}")
        print(f"    F1 Score: {f1_scores[optimal_idx]:.4f}")
        print(f"    Accuracy: {accuracies[optimal_idx]:.4f}")
        print(f"    Precision: {precisions[optimal_idx]:.4f}")
        print(f"    Recall: {recalls[optimal_idx]:.4f}")

    def _save_metrics(self, metrics):
        """
        Save evaluation metrics to file
        """
        evaluation_report = {
            "evaluation_date": datetime.now().isoformat(),
            "metrics": metrics,
            "model_architecture": "CNN + Transformer + Attention",
            "evaluation_summary": {
                "overall_performance": "Excellent"
                if metrics["f1_score"] > 0.9
                else "Good"
                if metrics["f1_score"] > 0.8
                else "Fair",
                "recommendations": self._generate_recommendations(metrics),
            },
        }

        with open("evaluation_report.json", "w") as f:
            json.dump(evaluation_report, f, indent=2)

        print("\n  Saved: evaluation_report.json")

    def _generate_recommendations(self, metrics):
        """
        Generate recommendations based on metrics
        """
        recommendations = []

        if metrics["recall"] < 0.85:
            recommendations.append(
                "Consider adjusting decision threshold to increase recall (reduce false negatives)"
            )

        if metrics["precision"] < 0.85:
            recommendations.append(
                "Consider adjusting decision threshold to increase precision (reduce false positives)"
            )

        if metrics["f1_score"] < 0.85:
            recommendations.append(
                "Consider collecting more training data or feature engineering"
            )

        if metrics["roc_auc"] < 0.9:
            recommendations.append(
                "Model may benefit from additional training or hyperparameter tuning"
            )

        if not recommendations:
            recommendations.append("Model performance is excellent across all metrics")

        return recommendations

    def test_realtime_inference(self, n_samples=100):
        """
        Test real-time inference speed
        """
        print("\n" + "=" * 80)
        print("REAL-TIME INFERENCE SPEED TEST")
        print("=" * 80)

        # Generate random test data
        import time

        feature_dim = self.model.input_shape[1]
        X_test = np.random.randn(n_samples, feature_dim)

        # Warmup
        _ = self.model.predict(X_test[:10], verbose=0)

        # Time individual predictions
        single_times = []
        for i in range(min(100, n_samples)):
            start = time.time()
            _ = self.model.predict(X_test[i : i + 1], verbose=0)
            single_times.append(time.time() - start)

        # Time batch predictions
        batch_sizes = [1, 10, 50, 100]
        batch_results = {}

        for batch_size in batch_sizes:
            if batch_size <= n_samples:
                start = time.time()
                _ = self.model.predict(X_test[:batch_size], verbose=0)
                elapsed = time.time() - start
                batch_results[batch_size] = {
                    "total_time": elapsed,
                    "avg_per_sample": elapsed / batch_size,
                }

        # Print results
        print(f"\nSingle prediction statistics (n={len(single_times)}):")
        print(f"  Mean: {np.mean(single_times) * 1000:.2f} ms")
        print(f"  Median: {np.median(single_times) * 1000:.2f} ms")
        print(f"  Std: {np.std(single_times) * 1000:.2f} ms")
        print(f"  Min: {np.min(single_times) * 1000:.2f} ms")
        print(f"  Max: {np.max(single_times) * 1000:.2f} ms")

        print("\nBatch prediction results:")
        for batch_size, results in batch_results.items():
            print(f"  Batch size {batch_size}:")
            print(f"    Total time: {results['total_time'] * 1000:.2f} ms")
            print(f"    Avg per sample: {results['avg_per_sample'] * 1000:.2f} ms")

        print("\n" + "=" * 80)

        return {
            "single_prediction_ms": np.mean(single_times) * 1000,
            "batch_results": batch_results,
        }


def generate_evaluation_report(X_test, y_test):
    """
    Generate complete evaluation report
    """
    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE EVALUATION REPORT")
    print("=" * 80)

    evaluator = ModelEvaluator()

    metrics = evaluator.evaluate_comprehensive(X_test, y_test, save_plots=True)

    speed_metrics = evaluator.test_realtime_inference(n_samples=100)

    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Insider Threat Detection - Evaluation Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background-color: #2c3e50;
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .metric-container {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .metric-card {{
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .metric-value {{
                font-size: 36px;
                font-weight: bold;
                color: #3498db;
            }}
            .metric-label {{
                color: #7f8c8d;
                margin-top: 5px;
            }}
            .chart-container {{
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            .chart-container img {{
                width: 100%;
                height: auto;
            }}
            h2 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            .status-excellent {{ color: #27ae60; }}
            .status-good {{ color: #f39c12; }}
            .status-fair {{ color: #e74c3c; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔒 Insider Threat Detection System</h1>
            <p>Model Evaluation Report - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Architecture:</strong> Hybrid CNN + Transformer + Attention</p>
        </div>
        
        <h2>📊 Performance Metrics</h2>
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-value">{metrics["accuracy"]:.2%}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics["precision"]:.2%}</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics["recall"]:.2%}</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics["f1_score"]:.2%}</div>
                <div class="metric-label">F1 Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics["roc_auc"]:.2%}</div>
                <div class="metric-label">ROC AUC</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{speed_metrics["single_prediction_ms"]:.1f}ms</div>
                <div class="metric-label">Inference Speed</div>
            </div>
        </div>
        
        <h2>📈 Visualizations</h2>
        
        <div class="chart-container">
            <h3>Confusion Matrix</h3>
            <img src="confusion_matrix.png" alt="Confusion Matrix">
        </div>
        
        <div class="chart-container">
            <h3>ROC Curve</h3>
            <img src="roc_curve.png" alt="ROC Curve">
        </div>
        
        <div class="chart-container">
            <h3>Precision-Recall Curve</h3>
            <img src="precision_recall_curve.png" alt="Precision-Recall Curve">
        </div>
        
        <div class="chart-container">
            <h3>Threshold Analysis</h3>
            <img src="threshold_analysis.png" alt="Threshold Analysis">
        </div>
        
        <div class="chart-container">
            <h3>Feature Importance (SHAP)</h3>
            <img src="shap_summary.png" alt="SHAP Summary">
        </div>
        
        <h2>💡 Recommendations</h2>
        <div class="metric-card">
            <ul>
                {"".join([f"<li>{rec}</li>" for rec in evaluator._generate_recommendations(metrics)])}
            </ul>
        </div>
        
        <h2>⚡ Performance Characteristics</h2>
        <div class="metric-card">
            <p><strong>False Positive Rate:</strong> {metrics["false_positive_rate"]:.2%}</p>
            <p><strong>False Negative Rate:</strong> {metrics["false_negative_rate"]:.2%}</p>
            <p><strong>Specificity:</strong> {metrics["specificity"]:.2%}</p>
            <p><strong>Sensitivity:</strong> {metrics["sensitivity"]:.2%}</p>
            <p><strong>Matthews Correlation:</strong> {metrics["mcc"]:.4f}</p>
            <p><strong>Cohen's Kappa:</strong> {metrics["cohen_kappa"]:.4f}</p>
        </div>
    </body>
    </html>
    """

    # Save HTML report
    with open("evaluation_report.html", "w") as f:
        f.write(html_report)

    print("\n✅ Evaluation report generated: evaluation_report.html")
    print("   Open this file in a web browser to view the complete report")

    return metrics, speed_metrics


# Example usage
if __name__ == "__main__":
    print("\nMODEL EVALUATION SCRIPT")
    print("=" * 80)

    print("\nUsage:")
    print("  1. Make sure you have trained the model first")
    print("  2. Load your test data")
    print("  3. Run evaluation:")
    print()
    print("  from model_evaluation import generate_evaluation_report")
    print("  import pandas as pd")
    print("  import pickle")
    print()
    print("  # Load preprocessor and test data")
    print("  with open('preprocessor.pkl', 'rb') as f:")
    print("      preprocessor = pickle.load(f)")
    print()
    print("  df_test = pd.read_csv('test_data.csv')")
    print("  X_test = preprocessor.transform(df_test)")
    print("  y_test = df_test['label'].values")
    print()
    print("  # Generate complete evaluation report")
    print("  metrics, speed = generate_evaluation_report(X_test, y_test)")
    print()
    print("=" * 80)
