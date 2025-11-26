"""
Dataset 2 Analysis – Refactored to match the concise teammate style.
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
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
    fig.savefig(path, dpi=300, bbox_inches='tight')
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
                'class 0': '0',
                'class 1': '1',
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
            raise ValueError("Call clean_data() before plotting correlations.")

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

        ax.set_title("Dataset 2 – Correlation Matrix")
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
        self._train_with_settings('Logistic Regression', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), scaled=True)
        self._train_with_settings('KNN', KNeighborsClassifier(n_neighbors=7), scaled=True)
        self._train_with_settings('SVM (RBF)', SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE), scaled=True)
        self._train_with_settings('Random Forest', RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE), scaled=False)
        self._train_with_settings('Gradient Boosting', GradientBoostingClassifier(random_state=RANDOM_STATE), scaled=False)
        self._train_with_settings('Naive Bayes', GaussianNB(), scaled=True)

    def _train_with_settings(self, name: str, model, scaled: bool) -> None:
        if scaled:
            X_train = self.data_processor.X_train_scaled
            X_test = self.data_processor.X_test_scaled
        else:
            X_train = self.data_processor.X_train
            X_test = self.data_processor.X_test

        model.fit(X_train, self.data_processor.y_train)
        predictions = model.predict(X_test)
        self.results[name] = self._compute_metrics(model, predictions, scaled)

    def _compute_metrics(self, model, predictions: np.ndarray, scaled: bool) -> dict[str, object]:
        y_test = self.data_processor.y_test
        return {
            'model': model,
            'predictions': predictions,
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall': recall_score(y_test, predictions, zero_division=0),
            'f1': f1_score(y_test, predictions, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, predictions),
            'requires_scaling': scaled,
        }

    def get_feature_importance(self, classifier_name: str = 'Random Forest') -> np.ndarray | None:
        model = self.results.get(classifier_name, {}).get('model')
        if model is None:
            return None
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        if hasattr(model, 'coef_'):
            coefs = model.coef_
            return np.mean(np.abs(coefs), axis=0)
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

        best_name, best_result = self.get_best_classifier()
        print(f"\nBest classifier: {best_name} | Accuracy: {best_result['accuracy']:.4f}")

    def get_best_classifier(self) -> tuple[str, dict[str, object]]:
        return max(self.results.items(), key=lambda x: x[1]['accuracy'])

    def plot_confusion_matrices(self, filename_prefix: str) -> None:
        n = len(self.results)
        cols = min(3, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.array(axes).reshape(-1)

        for ax, (name, result) in zip(axes, self.results.items()):
            cm = result['confusion_matrix']
            ax.imshow(cm, cmap='Greens', aspect='auto')
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=text_color)
            ax.set_title(f"{name}\nAcc: {result['accuracy']:.3f}")
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Class 0', 'Class 1'])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Class 0', 'Class 1'])

        for ax in axes[len(self.results):]:
            ax.axis('off')

        fig.tight_layout()
        _save_plot(fig, f"{filename_prefix}_combined.png")

    def plot_comparison(self, filename: str) -> None:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        classifier_names = list(self.results.keys())

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(classifier_names))
        width = 0.18
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for idx, metric in enumerate(metrics):
            values = [self.results[name][metric] for name in classifier_names]
            ax.bar(x + idx * width, values, width, label=metric.capitalize(), color=colors[idx])

        ax.set_xlabel('Classifier')
        ax.set_ylabel('Score')
        ax.set_title('Dataset 2 – Classifier Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(classifier_names, rotation=20, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        _save_plot(fig, filename)

    def plot_feature_importance(self, filename: str, classifier_name: str = 'Random Forest') -> None:
        importance = self.classifier.get_feature_importance(classifier_name)
        if importance is None:
            print(f'{classifier_name} does not provide feature importances.')
            return

        features = self.classifier.data_processor.df.drop(columns=['label']).columns
        x_pos = np.arange(len(features))
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x_pos, importance, color='#2ca02c')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.set_ylabel('Importance Score')
        ax.set_title(f'Feature Importances – {classifier_name}')
        fig.tight_layout()
        _save_plot(fig, filename)

    def plot_learning_curve(self, classifier_name: str, filename: str) -> None:
        features = self.classifier.data_processor.df.drop(columns=['label'])
        y = self.classifier.data_processor.df['label'].values
        best_meta = self.results[classifier_name]
        estimator = clone(best_meta['model'])

        if best_meta['requires_scaling']:
            estimator = Pipeline([
                ('scaler', StandardScaler()),
                ('model', estimator),
            ])

        train_sizes, train_scores, test_scores = learning_curve(
            estimator,
            features,
            y,
            train_sizes=np.linspace(0.1, 1.0, 8),
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train Score')
        ax.plot(train_sizes, test_scores.mean(axis=1), 's-', label='CV Score')
        ax.fill_between(
            train_sizes,
            train_scores.mean(axis=1) - train_scores.std(axis=1),
            train_scores.mean(axis=1) + train_scores.std(axis=1),
            alpha=0.2,
        )
        ax.fill_between(
            train_sizes,
            test_scores.mean(axis=1) - test_scores.std(axis=1),
            test_scores.mean(axis=1) + test_scores.std(axis=1),
            alpha=0.2,
        )
        ax.set_xlabel('Training Samples')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Learning Curve – {classifier_name}')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        _save_plot(fig, filename)


def main() -> None:
    processor = DataProcessor('dataset_2.csv')
    processor.clean_data()
    processor.plot_correlation_matrix('dataset2_correlation_matrix.png')
    processor.split_and_scale()

    classifier = Classifier(processor)
    classifier.train_all()

    evaluator = Evaluator(classifier)
    evaluator.print_summary()
    evaluator.plot_confusion_matrices('dataset2_confusion_matrix')
    evaluator.plot_comparison('dataset2_classifier_comparison.png')
    evaluator.plot_feature_importance('dataset2_feature_importance.png')

    best_name, _ = evaluator.get_best_classifier()
    evaluator.plot_learning_curve(best_name, 'dataset2_learning_curve.png')


if __name__ == '__main__':
    main()
