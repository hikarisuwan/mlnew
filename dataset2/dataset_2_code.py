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

# main execution function for the dataset 2 pipeline
# args: None
# returns: None
# stores: generates output files in the 'outputs_ds2' directory
def main() -> None:
    # construct the absolute path to the dataset csv file
    csv_path = os.path.join(current_dir, 'dataset_2.csv')
    
    # run the full analysis pipeline using specific settings for dataset 2
    # dataset 2 is smaller, so we enable 'impute_missing' to preserve data points rather than dropping rows
    # we enable learning curves to check for overfitting/underfitting given the smaller sample size
    run_full_analysis(
        dataset_path=csv_path,
        output_dir_name='outputs_ds2',
        run_feature_importance=False, 
        run_learning_curve=True,
        impute_missing=True 
    )

if __name__ == '__main__':
    main()
