import sys
import os 

# we modify sys.path to ensure 'ml_classes' imports from any working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from ml_classes import run_full_analysis

def main() -> None:
    csv_path = os.path.join(current_dir, 'dataset_2.csv')
    
    # run pipeline for dataset 2 
    run_full_analysis(
        dataset_path=csv_path,
        output_dir_name='outputs_ds2',
        run_feature_importance=False, 
        run_learning_curve=True,
        impute_missing=True
    )

if __name__ == '__main__':
    main()