"""
Test Script - Verify Setup and Dependencies
"""

import sys


def test_imports():
    """Test that all required packages are installed."""
    print("Testing package imports...")
    
    required_packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('sklearn', 'Scikit-learn')
    ]
    
    failed = []
    for package, name in required_packages:
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {name:15s} {version}")
        except ImportError:
            print(f"  ✗ {name:15s} NOT INSTALLED")
            failed.append(name)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Run: source venv/bin/activate && pip install -r requirements.txt")
        return False
    
    print("\n✅ All required packages are installed!")
    return True


def test_datasets():
    """Test that datasets are available and readable."""
    print("\nTesting dataset access...")
    
    import os
    import pandas as pd
    
    datasets = [
        ('dataset_1.csv', 'Alloy Conductivity Dataset'),
        ('dataset_2.csv', 'Unknown Materials Dataset')
    ]
    
    failed = []
    for filename, description in datasets:
        if not os.path.exists(filename):
            print(f"  ✗ {description:30s} NOT FOUND ({filename})")
            failed.append(filename)
            continue
        
        try:
            df = pd.read_csv(filename)
            print(f"  ✓ {description:30s} {df.shape[0]:4d} samples, {df.shape[1]:2d} columns")
        except Exception as e:
            print(f"  ✗ {description:30s} ERROR: {e}")
            failed.append(filename)
    
    if failed:
        print(f"\n❌ Dataset issues: {', '.join(failed)}")
        return False
    
    print("\n✅ All datasets are accessible!")
    return True


def test_custom_classes():
    """Test that custom ML classes can be imported."""
    print("\nTesting custom ML classes...")
    
    try:
        from ml_classes import Preprocessor, Classifier, Evaluator
        print("  ✓ Preprocessor class")
        print("  ✓ Classifier class")
        print("  ✓ Evaluator class")
        print("\n✅ All custom classes can be imported!")
        return True
    except Exception as e:
        print(f"\n❌ Error importing custom classes: {e}")
        return False


def main():
    print("=" * 60)
    print("  SETUP VERIFICATION TEST")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test datasets
    if not test_datasets():
        all_passed = False
    
    # Test custom classes
    if not test_custom_classes():
        all_passed = False
    
    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("  ✅ ALL TESTS PASSED!")
        print("  You're ready to run the analysis!")
        print("\n  Run: python run_all.py")
    else:
        print("  ❌ SOME TESTS FAILED")
        print("  Please fix the issues above before proceeding.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())





