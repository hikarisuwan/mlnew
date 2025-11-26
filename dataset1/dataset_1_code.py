import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from ml_classes import DataProcessor, Classifier, Evaluator

def main() -> None:
    print("=== DATASET 1 ANALYSIS ===")
    
    csv_path = os.path.join(current_dir, 'dataset_1.csv')

    # 1. Process
    processor = DataProcessor(csv_path)
    processor.clean_data()
    processor.plot_correlation_matrix('dataset1_correlation_matrix.png')
    processor.split_and_scale()

    # 2. Train - ONLY specific models
    classifier = Classifier(processor)
    classifier.train_models(['Logistic Regression', 'Random Forest', 'KNN'])

    # 3. Evaluate
    evaluator = Evaluator(classifier)
    evaluator.print_summary()
    evaluator.plot_confusion_matrices('dataset1_confusion_matrix')
    evaluator.plot_comparison('dataset1_classifier_comparison.png')
    
    # 4. Feature Importance (Cost Reduction)
    evaluator.plot_feature_importance('dataset1_feature_importance.png')

if __name__ == '__main__':
    main()