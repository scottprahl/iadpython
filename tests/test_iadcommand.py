import sys
import os
import unittest
from unittest.mock import patch

# Calculate the path to the iadpython package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
iadpython_package_dir = os.path.join(parent_dir, 'iadpython')

# Append the iadpython package directory to sys.path
sys.path.append(iadpython_package_dir)

# Now you can import your script
import iadcommand

class TestCommandLineArgs(unittest.TestCase):
    
    def test_validator_01_valid(self):
        self.assertEqual(iadcommand.validator_01("0.5"), 0.5)

    def test_validator_01_invalid(self):
        with self.assertRaises(ValueError):
            iadcommand.validator_01("-1")

    def test_validator_11_valid(self):
        self.assertEqual(iadcommand.validator_11("-0.5"), -0.5)

    def test_validator_11_invalid(self):
        with self.assertRaises(ValueError):
            iadcommand.validator_11("2")

    def test_validator_positive_valid(self):
        self.assertEqual(iadcommand.validator_positive("10"), 10.0)

    def test_validator_positive_invalid(self):
        with self.assertRaises(ValueError):
            iadcommand.validator_positive("-5")

class TestIadCommand(unittest.TestCase):

    def test_valid_arguments(self):
        test_args = ['iadcommand.py', '-a', '0.5', '-b', '0.3', '-r', '0.5']
        with patch('sys.argv', test_args):
            iadcommand.main()

if __name__ == '__main__':
    unittest.main()
