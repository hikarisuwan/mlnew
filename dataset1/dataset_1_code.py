from ml_classes import DataProcessor, Classifier, Evaluator

def main() -> None:
    
    # 1. Process
    processor = DataProcessor('dataset1/dataset_1.csv')
    processor.clean_data()
    processor.plot_correlation_matrix('dataset1_correlation_matrix.png')
    processor.split_and_scale()

    # 2. Train -  Logistic Regression, Random Forest, and KNN 
    classifier = Classifier(processor)
    classifier.train_models(['Logistic Regression', 'Random Forest', 'KNN'])

    # 3. Evaluate
    evaluator = Evaluator(classifier)
    evaluator.print_summary()
    
    # Generates the combined matrix file
    evaluator.plot_confusion_matrices('dataset1_confusion_matrix')
    evaluator.plot_comparison('dataset1_classifier_comparison.png')
    
    # 4. Feature Importance for Cost Reduction
    evaluator.plot_feature_importance('dataset1_feature_importance.png')

if __name__ == '__main__':
    main()