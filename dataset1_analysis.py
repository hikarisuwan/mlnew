"""
Dataset 1 Analysis – Concise classification workflow inspired by teammate's style.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 67


def _save_plot(fig: plt.Figure, filename: str) -> None:
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {path}")
    plt.show(block=False)
    plt.close(fig)


class DataProcessor:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df: pd.DataFrame | None = None
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None
        self.X_train_scaled: np.ndarray | None = None
        self.X_test_scaled: np.ndarray | None = None
        self.scaler: StandardScaler | None = None

    def clean_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.filepath)
        df = df.dropna()
        labels = df['label']
        if labels.dtype == 'object':
            labels = labels.str.strip().str.lower().replace({
                'non-conductive': '0',
                'conductive': '1',
            })
        df['label'] = pd.to_numeric(labels, errors='coerce')
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)

        cleaned_path = Path(self.filepath).with_name(Path(self.filepath).stem + '_cleaned.csv')
        df.to_csv(cleaned_path, index=False)
        print(f"Cleaned data saved to {cleaned_path}")

        self.df = df
        return df

    def plot_correlation_matrix(self, filename: str) -> None:
        if self.df is None:
            raise ValueError("Data must be cleaned before plotting correlations.")

        corr = self.df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)

        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)

        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black')

        ax.set_title("Dataset 1 – Correlation Matrix")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        _save_plot(fig, filename)

    def split_and_scale(self, test_size: float = 0.2) -> None:
        if self.df is None:
            raise ValueError("Clean the data before splitting.")

        X = self.df.drop(columns=['label'])
        y = self.df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")


class Classifier:
    def __init__(self, data_processor: DataProcessor):
        self.data_processor = data_processor
        self.results: dict[str, dict[str, object]] = {}

    def train_all(self) -> None:
        self._train_logistic_regression()
        self._train_knn()
        self._train_random_forest()

    def _train_logistic_regression(self) -> None:
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(self.data_processor.X_train_scaled, self.data_processor.y_train)
        predictions = model.predict(self.data_processor.X_test_scaled)
        self.results['Logistic Regression'] = self._compute_metrics(model, predictions)

    def _train_knn(self) -> None:
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(self.data_processor.X_train_scaled, self.data_processor.y_train)
        predictions = model.predict(self.data_processor.X_test_scaled)
        self.results['KNN'] = self._compute_metrics(model, predictions)

    def _train_random_forest(self) -> None:
        model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
        model.fit(self.data_processor.X_train, self.data_processor.y_train)
        predictions = model.predict(self.data_processor.X_test)
        self.results['Random Forest'] = self._compute_metrics(model, predictions)

    def _compute_metrics(self, model, predictions: np.ndarray) -> dict[str, object]:
        y_test = self.data_processor.y_test
        return {
            'model': model,
            'predictions': predictions,
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall': recall_score(y_test, predictions, zero_division=0),
            'f1': f1_score(y_test, predictions, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, predictions),
        }

    def get_feature_importance(self, classifier_name: str = 'Random Forest') -> np.ndarray | None:
        model = self.results[classifier_name]['model']
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        if hasattr(model, 'coef_'):
            return np.abs(model.coef_[0])
        return None


class Evaluator:
    def __init__(self, classifier: Classifier):
        self.classifier = classifier
        self.results = classifier.results

    def print_summary(self) -> None:
        print("\nClassifier performance comparison\n")
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                'Classifier': name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Precision': f"{result['precision']:.4f}",
                'Recall': f"{result['recall']:.4f}",
                'F1-Score': f"{result['f1']:.4f}",
            })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        best_classifier = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest classifier: {best_classifier[0]} | Accuracy: {best_classifier[1]['accuracy']:.4f}")

    def plot_confusion_matrices(self, filename_prefix: str) -> None:
        fig, axes = plt.subplots(1, len(self.results), figsize=(5 * len(self.results), 4))
        if len(self.results) == 1:
            axes = [axes]

        for ax, (name, result) in zip(axes, self.results.items()):
            cm = result['confusion_matrix']
            ax.imshow(cm, cmap='Blues', aspect='auto')
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
            ax.set_title(f"{name}\nAcc: {result['accuracy']:.3f}")
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Non-Conductive', 'Conductive'])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Non-Conductive', 'Conductive'])

        fig.tight_layout()
        _save_plot(fig, f"{filename_prefix}_combined.png")

    def plot_comparison(self, filename: str) -> None:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        classifier_names = list(self.results.keys())

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(classifier_names))
        width = 0.2
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for idx, metric in enumerate(metrics):
            values = [self.results[name][metric] for name in classifier_names]
            ax.bar(x + idx * width, values, width, label=metric.capitalize(), color=colors[idx])

        ax.set_xlabel('Classifier')
        ax.set_ylabel('Score')
        ax.set_title('Classifier Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(classifier_names)
        ax.legend()
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        _save_plot(fig, filename)

    def plot_feature_importance(self, filename: str) -> None:
        importance = self.classifier.get_feature_importance('Random Forest')
        if importance is None:
            print('Random Forest did not expose feature importances.')
            return

        features = self.classifier.data_processor.df.drop(columns=['label']).columns
        fig, ax = plt.subplots(figsize=(12, 6))
        x_pos = np.arange(len(features))
        ax.bar(x_pos, importance, color='#2ca02c')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.set_ylabel('Importance Score')
        ax.set_title('Feature Importances – Random Forest')
        fig.tight_layout()
        _save_plot(fig, filename)


def main() -> None:
    processor = DataProcessor('dataset_1.csv')
    processor.clean_data()
    processor.plot_correlation_matrix('dataset1_correlation_matrix.png')
    processor.split_and_scale()

    classifier = Classifier(processor)
    classifier.train_all()

    evaluator = Evaluator(classifier)
    evaluator.print_summary()
    evaluator.plot_confusion_matrices('dataset1_confusion_matrix')
    evaluator.plot_comparison('dataset1_classifier_comparison.png')
    evaluator.plot_feature_importance('dataset1_feature_importance.png')


if __name__ == '__main__':
    main()
