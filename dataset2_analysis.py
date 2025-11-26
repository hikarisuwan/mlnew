
from ml_classes import DataProcessor, Classifier, Evaluator

def main() -> None:
    print("=== DATASET 2 ANALYSIS ===")
    
    # 1. Process
    processor = DataProcessor('dataset_2.csv')
    processor.clean_data()
    processor.plot_correlation_matrix('dataset2_correlation_matrix.png')
    processor.split_and_scale()

    # 2. Train
    classifier = Classifier(processor)
    classifier.train_all()

    # 3. Evaluate
    evaluator = Evaluator(classifier)
    evaluator.print_summary()
    evaluator.plot_confusion_matrices('dataset2_confusion_matrix')
    evaluator.plot_comparison('dataset2_classifier_comparison.png')
    
    # 4. Critical: Learning Curve for Sample Size
    best_name, _ = evaluator.get_best_classifier()
    print(f"\nGenerating Learning Curve for {best_name}...")
    evaluator.plot_learning_curve(best_name, 'dataset2_learning_curve.png')

if __name__ == '__main__':
    main()