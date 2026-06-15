import unittest
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚀 Running Organic CIM Simulation Platform Unit Tests...")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir='tests', pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("=" * 60)
    if result.wasSuccessful():
        print("✅ All unit tests passed successfully!")
        sys.exit(0)
    else:
        print("❌ Unit test execution failed!")
        sys.exit(1)
