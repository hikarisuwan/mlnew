import sys
import os 

# we added the parent directory to sys.path to import 'ml_classes'. import os is so that you can run command from any directory 
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
        run_learning_curve=True
    )

if __name__ == '__main__':
    main()
