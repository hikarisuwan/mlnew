from ml_classes import DataProcessor, Classifier, Evaluator

def main() -> None:
    
    # 1. Process
    processor = DataProcessor('dataset2/dataset_2.csv')
    processor.clean_data()
    processor.plot_correlation_matrix('dataset2_correlation_matrix.png')
    processor.split_and_scale()

    # 2. Train - Trains all available models 
    classifier = Classifier(processor)
    classifier.train_models() 

    # 3. Evaluate
    evaluator = Evaluator(classifier)
    evaluator.print_summary()
    
    # Generates the combined matrix file
    evaluator.plot_confusion_matrices('dataset2_confusion_matrix')
    evaluator.plot_comparison('dataset2_classifier_comparison.png')
    
    # 4. Learning Curve 
    best_name, _ = evaluator.get_best_classifier()
    print(f"\nGenerating Learning Curve for {best_name}...")
    evaluator.plot_learning_curve(best_name, 'dataset2_learning_curve.png')

if __name__ == '__main__':
    main()