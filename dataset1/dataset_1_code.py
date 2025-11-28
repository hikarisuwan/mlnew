import sys
import os

# we modify the parent directory to sys.path to import 'ml_classes'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from ml_classes import run_full_analysis

def main() -> None:
    csv_path = os.path.join(current_dir, 'dataset_1.csv')
    
    # run pipeline for dataset 1
    run_full_analysis(
        dataset_path=csv_path,
        output_dir_name='outputs_ds1',
        model_list=['Logistic Regression', 'Random Forest', 'KNN'],
        run_feature_importance=True,  
        run_learning_curve=False      
    )

if __name__ == '__main__':
    main()