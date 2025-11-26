
import sys

def test_imports():
    print("Testing package imports...")
    required = ['numpy', 'pandas', 'matplotlib', 'sklearn']
    failed = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} NOT INSTALLED")
            failed.append(pkg)
    if failed: return False
    print("\nAll required packages are installed!")
    return True

def test_custom_classes():
    print("\nTesting custom ML classes...")
    try:
        from ml_classes import DataProcessor, Classifier, Evaluator
        print("  ✓ DataProcessor class")
        print("  ✓ Classifier class")
        print("  ✓ Evaluator class")
        print("\nAll custom classes can be imported!")
        return True
    except Exception as e:
        print(f"\nError importing custom classes: {e}")
        return False

def main():
    print("="*40 + "\n  SETUP VERIFICATION TEST\n" + "="*40)
    if test_imports() and test_custom_classes():
        print("\n  ALL TESTS PASSED! Run: python run_all.py")
        return 0
    else:
        print("\n  SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())