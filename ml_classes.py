from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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

# does loading, cleaning and preprocessing of the dataset
class DataProcessor:

    def __init__(self, filepath: str, random_state: int = 67):
        self.filepath = filepath
        self.random_state = random_state
        self.df: pd.DataFrame | None = None
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None
        self.X_train_scaled: np.ndarray | None = None
        self.X_test_scaled: np.ndarray | None = None
        self.scaler: StandardScaler | None = None
        self.use_imputation: bool = False
        
    # loads data, standardises labels and removes missing values for large dataset (dataset 1) or imputes them for small dataset (dataset 2)
    def clean_data(self, impute: bool = False) -> pd.DataFrame:
        self.use_imputation = impute
        df = pd.read_csv(self.filepath)
        
        # if we are not imputing, we drop rows with missing values 
        # if we are imputing, we leave them in to prevent data leakage.
        # imputation is handled later in split_and_scale using statistics derived only from the training set.
        if not impute:
            df = df.dropna()

        # standardise labels 
        labels = df.iloc[:, -1]
        if labels.dtype == 'object':
            # handle different label formats 
            labels = labels.str.strip().str.lower().replace({
                'non-conductive': '0', 
                'conductive': '1',
            })
        
        target_col = df.columns[-1]
        df[target_col] = pd.to_numeric(labels, errors='coerce')
        
        # we always drop rows where the TARGET is missing as we can't train/test on those
        df = df.dropna(subset=[target_col])
        df[target_col] = df[target_col].astype(int)
        
        if target_col != 'label':
            df = df.rename(columns={target_col: 'label'})

        cleaned_path = Path(self.filepath).with_name(Path(self.filepath).stem + '_cleaned.csv')
        df.to_csv(cleaned_path, index=False)

        self.df = df
        return df
        
    # generates a correlation matrix heatmap
    def plot_correlation_matrix(self, save_path: Path) -> None:
        if self.df is None:
            return

        corr = self.df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)

        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)

        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                val = corr.iloc[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=color)

        ax.set_title("Correlation Matrix")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # splits data into train/test sets and scales features
    def split_and_scale(self, test_size: float = 0.2) -> None:
        if self.df is None:
            return

        X = self.df.drop(columns=['label'])
        y = self.df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # apply imputation if requested
        if self.use_imputation:
            imputer = SimpleImputer(strategy='mean')
            X_train_imputed = imputer.fit_transform(X_train)
            X_test_imputed = imputer.transform(X_test)
            
            # reconstruct dataFrames
            X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns, index=X_test.index)

        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

# manages model training
class Classifier:
    
    def __init__(self, data_processor: DataProcessor, random_state: int = 67):
        self.dp = data_processor
        self.random_state = random_state
        self.results: dict[str, dict[str, object]] = {}
        
    # trains specified models or all models if None provided
    def train_models(self, model_names: list[str] | None = None) -> None:
        # we define a dictionary of models with flags for whether they need feature scaling
        all_models = {
            'Logistic Regression': (LogisticRegression(max_iter=2000, random_state=self.random_state), True),
            'KNN': (KNeighborsClassifier(n_neighbors=5), True),
            'Random Forest': (RandomForestClassifier(n_estimators=200, random_state=self.random_state), False),
            'SVM': (SVC(kernel='rbf', probability=True, random_state=self.random_state), True),
            'Gradient Boosting': (GradientBoostingClassifier(random_state=self.random_state), False),
            'Naive Bayes': (GaussianNB(), True)
        }

        models_to_train = {name: all_models[name] for name in model_names} if model_names else all_models

        for name, (model, use_scaled) in models_to_train.items():
            self._train_single_model(name, model, use_scaled)

    def _train_single_model(self, name: str, model, scaled: bool) -> None:
        X_train = self.dp.X_train_scaled if scaled else self.dp.X_train
        X_test = self.dp.X_test_scaled if scaled else self.dp.X_test

        model.fit(X_train, self.dp.y_train)
        predictions = model.predict(X_test)
        self.results[name] = self._compute_metrics(model, predictions)

    def _compute_metrics(self, model, predictions: np.ndarray) -> dict[str, object]:
        y_test = self.dp.y_test
        return {
            'model': model,
            'predictions': predictions,
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall': recall_score(y_test, predictions, zero_division=0),
            'f1': f1_score(y_test, predictions, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, predictions),
            'requires_scaling': isinstance(model, (LogisticRegression, KNeighborsClassifier, SVC, GaussianNB))
        }

    def get_feature_importance(self, classifier_name: str = 'Random Forest') -> np.ndarray | None:
        if classifier_name not in self.results:
            return None
        model = self.results[classifier_name]['model']
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        if hasattr(model, 'coef_'):
            return np.abs(model.coef_[0])
        return None

# generates plots for classifiers
class Evaluator:
   
    def __init__(self, classifier: Classifier, output_dir: Path):
        self.classifier = classifier
        self.results = classifier.results
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_plot(self, fig: plt.Figure, filename: str) -> None:
        path = self.output_dir / filename
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_confusion_matrices(self, filename_prefix: str) -> None:
        n = len(self.results)
        # we calculate the grid dimensions based on the number of trained models
        cols = min(3, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n == 1: axes = [axes]
        axes = np.array(axes).reshape(-1)

        for ax, (name, result) in zip(axes, self.results.items()):
            cm = result['confusion_matrix']
            ax.imshow(cm, cmap='Blues', aspect='auto')
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                           color='white' if cm[i,j] > cm.max()/2 else 'black')
            ax.set_title(f"{name}\nAcc: {result['accuracy']:.3f}")
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        for ax in axes[n:]: ax.axis('off')
        fig.tight_layout()
        self._save_plot(fig, f"{filename_prefix}_combined.png")

    def plot_comparison(self, filename: str) -> None:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        colors = ['#E53935', '#FF9800', '#FFEB3B', '#4CAF50'] 
        names = list(self.results.keys())
        fig, ax = plt.subplots(figsize=(14, 8)) 
        x = np.arange(len(names))
        width = 0.2
        
        for i, m in enumerate(metrics):
            vals = [self.results[n][m] for n in names]
            bars = ax.bar(x + i*width, vals, width, label=m.capitalize(), color=colors[i])
                # annotate each bar
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., 
                        height + 0.01, 
                        f'{height:.3f}', 
                        ha='center', va='bottom', fontsize=8) 
            
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels(names, rotation=20, ha='right')
        ax.legend(loc='lower right')
        ax.set_title('Classifier Performance Comparison')
        ax.set_ylim(0, 1.1) 
        fig.tight_layout()
        self._save_plot(fig, filename)

    def plot_feature_importance(self, filename: str, classifier_name: str = 'Random Forest') -> None:
        importance = self.classifier.get_feature_importance(classifier_name)
        if importance is None: 
            return

        features = self.classifier.dp.df.drop(columns=['label']).columns
        indices = np.argsort(importance)[::-1]
        sorted_feats = [features[i] for i in indices]
        sorted_imps = importance[indices]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(sorted_feats, sorted_imps, color='#2ca02c')

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, 
                    yval, 
                    f'{yval:.3f}', 
                    ha='center', va='bottom', fontsize=8) 

        ax.set_xticks(range(len(sorted_feats)))
        ax.set_xticklabels(sorted_feats, rotation=45, ha='right')
        ax.set_title(f'Feature Importances – {classifier_name}')
        fig.tight_layout()
        self._save_plot(fig, filename)

    def plot_learning_curve(self, classifier_name: str, filename: str) -> None:
        if classifier_name not in self.results: return
        
        best_meta = self.results[classifier_name]
        # we clone the estimator to ensure we start with a fresh, untrained model
        estimator = clone(best_meta['model'])
        
        # build pipeline to include imputation if needed
        steps = []
        if self.classifier.dp.use_imputation:
            # we include the imputer in the pipeline to prevent data leakage 
            steps.append(('imputer', SimpleImputer(strategy='mean')))
            
        if best_meta['requires_scaling']:
            steps.append(('scaler', StandardScaler()))
            
        steps.append(('model', estimator))
        
        pipeline = Pipeline(steps)
            
        X = self.classifier.dp.df.drop(columns=['label'])
        y = self.classifier.dp.df['label']
        
        train_sizes, train_scores, test_scores = learning_curve(
            pipeline, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 8), random_state=self.classifier.random_state
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train')
        ax.plot(train_sizes, test_scores.mean(axis=1), 'o-', label='CV')
        ax.axhline(y=0.7, color='r', linestyle='--', label='Target (0.7)')
        ax.set_title(f'Learning Curve – {classifier_name}')
        ax.legend()
        fig.tight_layout()
        self._save_plot(fig, filename)
        
def run_full_analysis(dataset_path: str, output_dir_name: str, model_list: list[str] | None = None,
                      run_feature_importance: bool = False, run_learning_curve: bool = False,
                      impute_missing: bool = False) -> None:
       
    # establish output paths relative to the dataset location
    base_dir = Path(dataset_path).parent
    output_dir = base_dir / output_dir_name
    
    #load and clean data
    processor = DataProcessor(dataset_path, random_state=67)
    processor.clean_data(impute=impute_missing)
    processor.plot_correlation_matrix(output_dir / 'correlation_matrix.png')
    processor.split_and_scale()

    #train and evaluate models
    classifier = Classifier(processor, random_state=67)
    classifier.train_models(model_list)

    #visualise results
    evaluator = Evaluator(classifier, output_dir)
    evaluator.plot_confusion_matrices('confusion_matrix')
    evaluator.plot_comparison('classifier_comparison.png')
    
    # conditional plotting for dataset 1 and 2
    if run_feature_importance:
        evaluator.plot_feature_importance('feature_importance.png', classifier_name='Random Forest')
        
    if run_learning_curve:
        if classifier.results:
            # automatic selection of the best model for the learning curve
            best_name, _ = max(classifier.results.items(), key=lambda x: x[1]['accuracy'])
            evaluator.plot_learning_curve(best_name, 'learning_curve.png')
