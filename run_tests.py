"""Simple test runner for the repository.
This runner calls test functions directly so pytest isn't required.
"""

import importlib
import sys

def run():
    failures = 0
    print('Running tests from tests/test_prep.py')
    try:
        tests = importlib.import_module('tests.test_prep')
        for name in dir(tests):
            if name.startswith('test_'):
                fn = getattr(tests, name)
                if callable(fn):
                    try:
                        print(f' - {name}...', end=' ')
                        fn()
                        print('OK')
                    except AssertionError as e:
                        print('FAIL')
                        print('   AssertionError:', e)
                        failures += 1
                    except Exception as e:
                        print('ERROR')
                        print('   Exception:', e)
                        failures += 1
    except Exception as e:
        print('Failed to import tests.test_prep:', e)
        sys.exit(2)

    if failures:
        print(f"\n{failures} test(s) failed.")
        sys.exit(1)
    else:
        print('\nAll tests passed')

if __name__ == '__main__':
    run()

