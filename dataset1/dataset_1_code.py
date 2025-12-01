import sys
import os

# get the directory of the current script to locate sibling files and parent directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# modify the system path to include the parent directory
# this allows us to import the 'ml_classes' module which is in the parent folder
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from ml_classes import run_full_analysis

# main execution function for the dataset 1 pipeline
# args: None
# returns: None
# stores: generates output files in the 'outputs_ds1' directory
def main() -> None:
    # construct the absolute path to the dataset csv file
    csv_path = os.path.join(current_dir, 'dataset_1.csv')
    
    # run the full analysis pipeline using specific settings for dataset 1
    # dataset 1 is large enough to not require imputation (rows with missing values are dropped)
    # we specifically request feature importance plots for this dataset
    run_full_analysis(
        dataset_path=csv_path,
        output_dir_name='outputs_ds1',
        model_list=['Logistic Regression', 'Random Forest', 'KNN'],
        run_feature_importance=True,  
        run_learning_curve=False      
    )

if __name__ == '__main__':
    main()
