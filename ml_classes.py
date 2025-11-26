"""
Machine Learning Classes for Materials Science Classification
Contains: Preprocessor, Classifier, and Evaluator classes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class Preprocessor:
    """
    Handles data loading, cleaning, normalization, and train-test splitting.
    """
    
    def __init__(self, filepath, test_size=0.2, random_state=67):
        """
        Initialize the preprocessor.
        
        Args:
            filepath: Path to the CSV file
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.filepath = filepath
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.original_size = None
        self.cleaned_size = None
        
    def load_data(self):
        """Load data from CSV file."""
        self.data = pd.read_csv(self.filepath)
        print(f"Data loaded: {self.data.shape[0]} samples, {self.data.shape[1]} columns")
        return self.data
    
    def explore_data(self):
        """Display basic information about the dataset."""
        print("\n=== Data Exploration ===")
        print(f"\nShape: {self.data.shape}")
        print(f"\nData types:\n{self.data.dtypes}")
        print(f"\nMissing values:\n{self.data.isnull().sum()}")
        print(f"\nBasic statistics:\n{self.data.describe()}")
        print(f"\nClass distribution:\n{self.data['label'].value_counts()}")
        
    def clean_data(self):
        """
        Clean the data by dropping rows with missing values and encoding labels.
        
        IMPORTANT: Instead of imputation, we drop entire rows containing missing values.
        This is critical for small datasets (~400 samples) to maintain data accuracy
        and model quality, as improper imputation can be detrimental.
        """
        # Store original size
        self.original_size = len(self.data)
        
        # Drop rows with any missing values
        self.data = self.data.dropna()
        self.cleaned_size = len(self.data)
        
        rows_dropped = self.original_size - self.cleaned_size
        
        print(f"\nData cleaning:")
        print(f"  Original samples: {self.original_size}")
        print(f"  Rows with missing values dropped: {rows_dropped}")
        print(f"  Remaining samples: {self.cleaned_size}")
        print(f"  Data retention: {self.cleaned_size/self.original_size*100:.1f}%")
        
        # Separate features and labels
        X = self.data.drop('label', axis=1)
        y = self.data['label']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Encode labels if they are strings
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
            print(f"\nLabel encoding: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        return X, y
    
    def split_and_scale(self, X, y):
        """
        Split data into train/test sets and apply feature scaling.
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\nData split: {len(self.X_train)} training samples, {len(self.X_test)} test samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def prepare_data(self):
        """
        Complete data preparation pipeline.
        """
        self.load_data()
        self.explore_data()
        X, y = self.clean_data()
        return self.split_and_scale(X, y)


class Classifier:
    """
    Handles training and prediction of classification models.
    """
    
    def __init__(self, model_type='logistic', random_state=67):
        """
        Initialize classifier.
        
        Args:
            model_type: Type of classifier ('logistic', 'random_forest', 'svm', etc.)
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._create_model()
        self.is_trained = False
        
    def _create_model(self):
        """Create the appropriate model based on model_type."""
        models = {
            'logistic': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'decision_tree': DecisionTreeClassifier(random_state=self.random_state),
            'svm': SVC(kernel='rbf', random_state=self.random_state),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'naive_bayes': GaussianNB(),
            'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state)
        }
        
        if self.model_type not in models:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return models[self.model_type]
    
    def train(self, X_train, y_train):
        """
        Train the classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"\nTraining {self.model_type} classifier...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training completed.")
        
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance scores if available.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Linear models - use absolute coefficient values
            importances = np.abs(self.model.coef_[0])
        else:
            print(f"Feature importance not available for {self.model_type}")
            return None
        
        return dict(zip(feature_names, importances))


class Evaluator:
    """
    Evaluates classifier performance and creates visualizations.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics = {}
    
    def plot_correlation_matrix(self, X, y, feature_names, title='Correlation Matrix', save_path=None):
        """
        Plot correlation matrix heatmap showing relationships between features and target.
        
        Args:
            X: Feature matrix (numpy array or pandas DataFrame)
            y: Target labels (numpy array or pandas Series)
            feature_names: List of feature names
            title: Plot title
            save_path: Path to save the figure (optional)
        """
        # Create DataFrame with features and target
        df = pd.DataFrame(X, columns=feature_names)
        df['label'] = y
        
        # Compute correlation matrix
        corr_matrix = df.corr()
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1, 
                    cbar_kws={"shrink": 0.8})
        
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved correlation matrix to {save_path}")
        
        plt.show()
        
        # Print correlation with target label
        print("\n=== Correlation with Target Label ===")
        label_corr = corr_matrix['label'].drop('label').sort_values(ascending=False)
        for feature, corr_value in label_corr.items():
            print(f"  {feature:30s}: {corr_value:7.4f}")
        
        return corr_matrix
        
    def compute_metrics(self, y_true, y_pred, model_name='Model'):
        """
        Compute classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model for storing metrics
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }
        
        self.metrics[model_name] = metrics
        
        print(f"\n=== {model_name} Performance ===")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, title='Confusion Matrix', labels=None, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            labels: Class labels for display
            save_path: Path to save the figure (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels if labels else ['Class 0', 'Class 1'],
                    yticklabels=labels if labels else ['Class 0', 'Class 1'])
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix to {save_path}")
        
        plt.show()
        
        return cm
    
    def plot_feature_importance(self, feature_importance_dict, title='Feature Importance', 
                                top_n=None, save_path=None):
        """
        Plot feature importance as a bar chart.
        
        Args:
            feature_importance_dict: Dictionary mapping features to importance scores
            title: Plot title
            top_n: Number of top features to display (None for all)
            save_path: Path to save the figure (optional)
        """
        if feature_importance_dict is None:
            print("No feature importance data available.")
            return
        
        # Sort by importance
        sorted_features = sorted(feature_importance_dict.items(), 
                                key=lambda x: x[1], reverse=True)
        
        if top_n:
            sorted_features = sorted_features[:top_n]
        
        features = [f[0] for f in sorted_features]
        importances = [f[1] for f in sorted_features]
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(features, importances, color='steelblue')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved feature importance plot to {save_path}")
        
        plt.show()
    
    def compare_classifiers(self, title='Classifier Comparison', save_path=None):
        """
        Create a comparison plot of different classifiers.
        
        Args:
            title: Plot title
            save_path: Path to save the figure (optional)
        """
        if not self.metrics:
            print("No metrics to compare. Train and evaluate models first.")
            return
        
        models = list(self.metrics.keys())
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
        
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metric_names):
            values = [self.metrics[model][metric] for model in models]
            ax.bar(x + i * width, values, width, label=metric.capitalize())
        
        ax.set_xlabel('Classifier', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved classifier comparison to {save_path}")
        
        plt.show()
    
    def plot_learning_curve(self, classifier, X, y, title='Learning Curve', 
                           cv=5, train_sizes=np.linspace(0.1, 1.0, 10), save_path=None):
        """
        Plot learning curve showing accuracy vs training set size.
        
        Args:
            classifier: Classifier object with a trained model
            X: Feature matrix
            y: Target labels
            title: Plot title
            cv: Number of cross-validation folds
            train_sizes: Array of training set sizes to evaluate
            save_path: Path to save the figure (optional)
        """
        print(f"\nGenerating learning curve... This may take a moment.")
        
        train_sizes_abs, train_scores, test_scores = learning_curve(
            classifier.model, X, y, cv=cv, train_sizes=train_sizes,
            scoring='accuracy', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes_abs, train_mean, 'o-', color='steelblue', 
                label='Training score', linewidth=2)
        
        plt.plot(train_sizes_abs, test_mean, 'o-', color='coral', 
                label='Cross-validation score', linewidth=2)
        
        # Add 70% accuracy reference line
        plt.axhline(y=0.7, color='green', linestyle='--', linewidth=2, 
                   label='70% Accuracy Target', alpha=0.7)
        
        plt.xlabel('Number of Training Samples', fontsize=12)
        plt.ylabel('Accuracy Score', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.ylim([0, 1.05])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved learning curve to {save_path}")
        
        plt.show()
        
        # Find minimum samples for 70% accuracy
        threshold_mask = test_mean >= 0.7
        if np.any(threshold_mask):
            min_samples = train_sizes_abs[threshold_mask][0]
            accuracy_at_min = test_mean[threshold_mask][0]
            print(f"\n✓ Minimum samples for 70% accuracy: {min_samples} samples")
            print(f"  Accuracy achieved: {accuracy_at_min:.4f}")
        else:
            print(f"\n✗ 70% accuracy not achieved with available data")
            print(f"  Maximum accuracy: {test_mean.max():.4f} at {train_sizes_abs[test_mean.argmax()]} samples")
        
        return train_sizes_abs, train_mean, test_mean




