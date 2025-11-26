
from ml_classes import DataProcessor, Classifier, Evaluator

def main() -> None:
    print("=== DATASET 1 ANALYSIS ===")
    
    # 1. Process
    processor = DataProcessor('dataset_1.csv')
    processor.clean_data()
    processor.plot_correlation_matrix('dataset1_correlation_matrix.png')
    processor.split_and_scale()

    # 2. Train
    classifier = Classifier(processor)
    classifier.train_all()

    # 3. Evaluate
    evaluator = Evaluator(classifier)
    evaluator.print_summary()
    evaluator.plot_confusion_matrices('dataset1_confusion_matrix')
    evaluator.plot_comparison('dataset1_classifier_comparison.png')
    
    # 4. Critical: Feature Importance for Cost Reduction
    print("\nVisualizing Feature Importance...")
    evaluator.plot_feature_importance('dataset1_feature_importance.png')

if __name__ == '__main__':
    main()