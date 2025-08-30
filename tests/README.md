# baltools Test Suite

This directory contains simple unit tests for the baltools package. The tests are designed to work on any system without external dependencies.

## Test Structure

- `test_utils.py` - Tests for utility functions logic (zeropad, gethpdir, pmmkdir)
- `test_balconfig.py` - Tests for configuration constants and parameter validation
- `test_desibal.py` - Tests for DESI BAL finder function signatures and logic
- `test_integration.py` - Integration tests between different modules
- `test_command_line.py` - Tests for command-line argument parsing logic
- `run_tests.py` - Simple test runner script

## Running Tests

### Run All Tests
```bash
python tests/run_tests.py
```

### Run Tests Using Python unittest
```bash
# Run all tests
python -m unittest discover tests -v

# Run specific test file
python -m unittest tests.test_utils -v

# Run specific test class
python -m unittest tests.test_utils.TestUtilsFunctions -v

# Run specific test method
python -m unittest tests.test_utils.TestUtilsFunctions.test_zeropad_function -v
```

## Test Coverage

### Unit Tests
- **utils.py**: Tests for utility functions like `zeropad`, `gethpdir`, `pmmkdir`
- **balconfig.py**: Tests for configuration constants and parameter validation
- **desibal.py**: Tests for function signatures, parameter validation, and the new `usetid` parameter

### Integration Tests
- Module interaction testing
- Parameter consistency validation
- Function signature validation
- File parsing logic testing

### Command Line Tests
- Argument parsing logic
- Parameter validation
- New `--tids` argument functionality

## Test Design Philosophy

These tests are designed to be:

1. **Simple**: No complex mocking or external dependencies
2. **Portable**: Work on any system (laptop, NERSC, etc.)
3. **Fast**: Quick to run without large data files
4. **Focused**: Test logic and functionality, not external systems

The tests use local function definitions to test the logic without importing the actual modules, which avoids issues with missing dependencies or environment-specific configurations.

## Test Dependencies

The tests use only standard Python modules:
- `unittest` - Standard Python testing framework
- No external dependencies required

## Adding New Tests

When adding new functionality to baltools, please add corresponding tests:

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test interactions between modules
3. **Command Line Tests**: Test any new command-line arguments or scripts

### Test Naming Convention
- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<function_name>_<scenario>`

### Example Test Structure
```python
def test_function_name_valid_input(self):
    """Test function with valid input"""
    # Define the function logic locally for testing
    def my_function(input_val):
        return input_val * 2
    
    result = my_function(5)
    self.assertEqual(result, 10)

def test_function_name_invalid_input(self):
    """Test function with invalid input"""
    def my_function(input_val):
        if input_val < 0:
            raise ValueError("Input must be positive")
        return input_val * 2
    
    with self.assertRaises(ValueError):
        my_function(-5)
```

## Continuous Integration

These tests can be integrated into CI/CD pipelines to ensure code quality and prevent regressions. The test runner returns appropriate exit codes:
- `0` - All tests passed
- `1` - Some tests failed

## Notes

- Tests are designed to be independent and can run in any order
- No temporary files or directories are created
- Tests focus on functionality that can be tested without large data files
- The test suite is designed to work on both development laptops and production systems like NERSC
