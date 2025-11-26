# Machine Learning Classification for Materials Science
## Computing Challenge 2025-2026 - Materials.AI.ML

This project implements machine learning classifiers for two materials science datasets, following a structured class-based approach with comprehensive analysis and visualizations.

## Project Overview

### Dataset 1: Alloy Conductivity Classification
- **Goal**: Predict whether alloy samples are conductive or non-conductive
- **Features**: 10 material properties (density, vacancy content, melting temperature, etc.)
- **Objective**: Identify most important features to reduce measurement costs

### Dataset 2: Unknown Material Classification
- **Goal**: Build the best possible classifier for material classification
- **Features**: 8 anonymous features
- **Objective**: Determine minimum datapoints required for 70% accuracy

## Project Structure

```
advanced python computing project 2025/
├── ml_classes.py                   # Core ML classes (Preprocessor, Classifier, Evaluator)
├── dataset1_analysis.py            # Dataset 1 analysis script
├── dataset2_analysis.py            # Dataset 2 analysis script
├── materials_classification.ipynb  # Jupyter notebook version
├── requirements.txt                # Python dependencies
├── dataset_1.csv                   # Alloy conductivity dataset
├── dataset_2.csv                   # Unknown materials dataset
├── DATASET1_REPORT.md              # Written report for Dataset 1
├── DATASET2_REPORT.md              # Written report for Dataset 2
├── SUBMISSION_GUIDE.md             # Submission instructions
└── README.md                       # This file
```

## Installation

1. Ensure you have Python 3.8+ installed
2. Install required packages:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## Usage

### Running the Analysis Scripts

Run the individual analysis scripts:

```bash
# Dataset 1 - Alloy Conductivity Classification
python dataset1_analysis.py

# Dataset 2 - Unknown Material Classification
python dataset2_analysis.py
```

Each script will:
- Load and clean the data
- Train multiple classifiers
- Generate visualizations in `outputs/` directory
- Print performance metrics

### Alternative: Jupyter Notebook

```bash
jupyter notebook materials_classification.ipynb
```

Then run all cells or step through interactively

## Code Architecture

The solution follows object-oriented design with three main classes:

### 1. Preprocessor
- Loads and explores data
- **Handles missing values by dropping entire rows** (maintains data accuracy for small datasets)
- Encodes categorical labels
- Splits data into training/testing sets
- Applies feature scaling (standardization)

### 2. Classifier
- Supports multiple classifier types:
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - Gradient Boosting
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
- Trains models on provided data
- Makes predictions
- Extracts feature importance (when available)

### 3. Evaluator
- Computes performance metrics (accuracy, precision, recall, F1-score)
- Generates correlation matrices and heatmaps
- Creates confusion matrices
- Plots feature importance
- Compares multiple classifiers
- Generates learning curves

## Key Results

### Dataset 1: Cost Reduction Strategy
- **Best Classifier**: Random Forest (100% accuracy)
- **Key Finding**: Band gap is a perfect predictor (94.53% importance)
- **Recommendation**: Measure only band gap (90% cost savings, 100% accuracy)
- **Deliverables**:
  - Feature importance visualization showing band gap dominance
  - Accuracy vs. number of features plot (100% with just 1 feature)
  - Confusion matrix (perfect classification)
  - Physical justification: band_gap = 0.0 → conductive, else non-conductive

### Dataset 2: Performance Optimization
- **Classifiers Tested**: Logistic Regression, Random Forest, SVM, KNN, Gradient Boosting, Naive Bayes
- **Analysis**: Comprehensive comparison and learning curve analysis
- **Deliverables**:
  - Confusion matrices for all classifiers
  - Classifier comparison chart
  - Learning curve showing accuracy vs. training size
  - Minimum sample requirements for 70% accuracy

## Visualizations Generated

1. **Correlation Matrices & Heatmaps**: Show relationships between features and target label
2. **Confusion Matrices**: Show true vs. predicted classifications
3. **Feature Importance Plot**: Ranks features by predictive power (Dataset 1)
4. **Classifier Comparison**: Bar chart comparing all metrics across models
5. **Learning Curves**: Show model performance vs. training data size
6. **Accuracy vs. Features**: Cost-benefit analysis (Dataset 1)

## Requirements Met

✓ **Implementation** (20/100 marks): Complete ML pipeline with data analysis, train-test splitting, training, and evaluation  
✓ **Code Structure** (20/100 marks): Organized into Preprocessor, Classifier, and Evaluator classes  
✓ **Scikit-learn Usage** (20/100 marks): Proper use of sklearn for models, metrics, preprocessing, and cross-validation  
✓ **Analysis & Recommendations** (40/100 marks): Comprehensive reports with quantitative justification

## Performance Metrics

All classifiers are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## Notes

- **Missing values are handled by dropping entire rows** (critical for small datasets to maintain accuracy)
- See `DATA_CLEANING_APPROACH.md` for detailed justification
- All results are reproducible (random seed = 42)
- Feature scaling is applied to improve model performance
- Cross-validation is used for learning curve generation
- Both text and numeric labels are properly encoded

## Author

Submitted for Computing Challenge 2025-2026

## License

Academic project - All rights reserved

