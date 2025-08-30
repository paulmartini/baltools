#!/usr/bin/env python
"""
Simple test runner for baltools package
"""

import unittest
import sys
import os

def main():
    """Main function for running tests"""
    # Add the parent directory to the path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'py'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return appropriate exit code
    if result.wasSuccessful():
        print("\n" + "=" * 50)
        print("All tests passed! ✅")
        sys.exit(0)
    else:
        print("\n" + "=" * 50)
        print("Some tests failed! ❌")
        sys.exit(1)

if __name__ == '__main__':
    main()
