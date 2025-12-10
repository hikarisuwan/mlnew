# ML analysis project

Code for analysing two datasets using common classifiers (Random forest, KNN etc).

Setup

You need Python installed. The scripts use these libraries:
- pandas
- numpy
- matplotlib
- scikit-learn

Install them with:
`pip install -r requirements.txt`

How to Run:

There are two separate scripts, one for each dataset. You can also run both at once.

Run everything: `python run_all.py`

Dataset 1: `python dataset1/dataset_1_code.py`
Results go to dataset1/outputs_ds2/

Dataset 2:`python dataset2/dataset_2_code.py`
Results go to dataset2/outputs_ds2/

File Structure:
```
.
├── dataset1/
│   ├── outputs_ds1/                # generated plots 
│   ├── dataset_1.csv               # input data for dataset 1
│   └── dataset_1_code.py           # runner script for dataset 1
├── dataset2/
│   ├── outputs_ds2/                # generated plots 
│   ├── dataset_2.csv               # input data for dataset 2
│   └── dataset_2_code.py           # runner script for dataset 2
├── ml_classes.py                   # main logic for cleaning, training and plotting
├── run_all.py                      # master script to run both analyses
├── requirements.txt                # list of python dependencies
├── .gitignore                      # git exclusion rules
└── README.md                       # project documentation
```
