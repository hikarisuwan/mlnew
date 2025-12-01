# ML Analysis Project

Code for analyzing two datasets using common classifiers (Random forest, KNN etc).

Setup

You need Python installed. The scripts use these libraries:
- pandas
- numpy
- matplotlib
- scikit-learn

Install them with:
pip install pandas numpy matplotlib scikit-learn 

How to Run:
There are two separate scripts, one for each dataset. Run them from the main project folder so the imports work correctly.

Dataset 1: python dataset1/dataset_1_code.py
Results go to dataset1/outputs_ds2/

Dataset 2:python dataset2/dataset_2_code.py
Results go to dataset2/outputs_ds2/

File Structure:
.
├── dataset1/
│   ├── outputs_ds1/ #gets generated upon running
│   │   ├── classifier_comparison.png
│   │   ├── confusion_matrix_combined.png
│   │   ├── correlation_matrix.png
│   │   └── feature_importance.png
│   ├── dataset_1.csv
│   └── dataset_1_code.py #runner script for dataset 1
├── dataset2/
│   ├── outputs_ds2/ #gets generated upon running 
│   │   ├── classifier_comparison.png
│   │   ├── confusion_matrix_combined.png
│   │   ├── correlation_matrix.png
│   │   └── learning_curve.png
│   ├── dataset_2.csv
│   └── dataset_2_code.py # runner script for dataset 2
├── ml_classes.py # main logic for cleaning, training and plotting
└── README.md




