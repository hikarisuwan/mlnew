
import os
import sys
from datetime import datetime


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main():
    print_header("MACHINE LEARNING CLASSIFICATION FOR MATERIALS SCIENCE")
    print("Materials.AI.ML - Computing Challenge 2025-2026")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    print("\n✓ Outputs directory created/verified")
    
    # Run Dataset 1 Analysis
    print_header("PART 1: ALLOY CONDUCTIVITY CLASSIFICATION")
    try:
        import dataset1_analysis
        dataset1_analysis.main()
        print("\n✅ Dataset 1 analysis completed successfully!")
    except Exception as e:
        print(f"\n❌ Error in Dataset 1 analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Run Dataset 2 Analysis
    print_header("PART 2: UNKNOWN MATERIAL CLASSIFICATION")
    try:
        import dataset2_analysis
        dataset2_analysis.main()
        print("\n✅ Dataset 2 analysis completed successfully!")
    except Exception as e:
        print(f"\n❌ Error in Dataset 2 analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    print_header("ANALYSIS COMPLETE")
    print("📊 Summary of Deliverables:")
    print("\n🗂️  Generated Files:")
    print("   outputs/dataset1_classifier_comparison.png")
    print("   outputs/dataset1_feature_importance.png")
    print("   outputs/dataset1_confusion_matrix.png")
    print("   outputs/dataset1_accuracy_vs_features.png")
    print("   outputs/dataset2_classifier_comparison.png")
    print("   outputs/dataset2_confusion_matrix_*.png (6 files)")
    print("   outputs/dataset2_learning_curve.png")
    
    print("\n📄 Written Reports:")
    print("   DATASET1_REPORT.md")
    print("   DATASET2_REPORT.md")
    
    print("\n💡 Next Steps:")
    print("   1. Review all visualizations in 'outputs/' directory")
    print("   2. Read the executive summary reports")
    print("   3. Check SUBMISSION_GUIDE.md for submission instructions")
    
    print(f"\n✅ All analyses completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)





