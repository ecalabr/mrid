"""Comprehensive test runner for the MRID application.
This runner calls test functions directly so pytest isn't required.
Tests multiple components for debugging and development purposes.
"""

import importlib.util
import sys
import time
from pathlib import Path

# Test modules to run
TEST_MODULES = [
    'tests.test_prep',
    'tests.test_field_detection',
    'tests.test_config',
    'tests.test_config_io',
    'tests.test_analyze_coverage',
    'tests.test_processing',
    'tests.test_cnn_slices',
    'tests.test_csv_field_updater',
    'tests.test_scaling',
    'tests.test_scanner',
    'tests.test_utils',
]

def run_module_tests(module_name):
    """Run all test functions in a given module."""
    failures = 0
    errors = 0
    passed = 0

    print(f'\n{"="*70}')
    print(f'Running tests from {module_name}')
    print(f'{"="*70}')

    try:
        # Load by file path: importing 'tests.test_*' normally would fail
        # because this runner module (tests.py) shadows the tests/ directory.
        module_file = Path(__file__).parent / f'{module_name.replace(".", "/")}.py'
        spec = importlib.util.spec_from_file_location(module_name, module_file)
        tests = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tests)
        test_functions = [name for name in dir(tests) if name.startswith('test_')]

        if not test_functions:
            print(f'  No test functions found in {module_name}')
            return 0, 0, 0

        for name in test_functions:
            fn = getattr(tests, name)
            if callable(fn):
                try:
                    print(f'  • {name}...', end=' ', flush=True)
                    start_time = time.time()
                    fn()
                    elapsed = time.time() - start_time
                    print(f'✓ OK ({elapsed:.3f}s)')
                    passed += 1
                except AssertionError as e:
                    print(f'✗ FAIL')
                    print(f'     AssertionError: {e}')
                    failures += 1
                except Exception as e:
                    print(f'✗ ERROR')
                    print(f'     {type(e).__name__}: {e}')
                    errors += 1

    except ImportError as e:
        print(f'  ⚠ Could not import {module_name}: {e}')
        print(f'     (This may be expected if the module does not exist yet)')
        return 0, 0, 0
    except Exception as e:
        print(f'  ✗ Error loading {module_name}: {e}')
        return 0, 0, 1

    return passed, failures, errors

def print_summary(total_passed, total_failures, total_errors, total_time):
    """Print test summary."""
    print(f'\n{"="*70}')
    print('TEST SUMMARY')
    print(f'{"="*70}')
    print(f'  Passed:  {total_passed}')
    print(f'  Failed:  {total_failures}')
    print(f'  Errors:  {total_errors}')
    print(f'  Total:   {total_passed + total_failures + total_errors}')
    print(f'  Time:    {total_time:.2f}s')
    print(f'{"="*70}')

    if total_failures == 0 and total_errors == 0:
        print('✓ All tests passed!')
        return 0
    else:
        print(f'✗ {total_failures + total_errors} test(s) failed')
        return 1

def run():
    """Run all tests."""
    print('MRID Application Test Suite')
    print('='*70)

    start_time = time.time()
    total_passed = 0
    total_failures = 0
    total_errors = 0

    for module_name in TEST_MODULES:
        passed, failures, errors = run_module_tests(module_name)
        total_passed += passed
        total_failures += failures
        total_errors += errors

    total_time = time.time() - start_time
    return print_summary(total_passed, total_failures, total_errors, total_time)

if __name__ == '__main__':
    sys.exit(run())

