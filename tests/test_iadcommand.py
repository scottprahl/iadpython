"""Tests for the command line wrapper."""
import sys
import os
import unittest
import argparse
from unittest.mock import patch

# Calculate the path to the iadpython package so pytest works from any directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
iadpython_package_dir = os.path.join(parent_dir, 'iadpython')
sys.path.append(iadpython_package_dir)

# Now we can import iadcommand
import iadcommand


class TestCommandLineArgs(unittest.TestCase):
    """Tests for validator functions."""

    def test_validator_01_valid(self):
        """Valid test that 01."""
        self.assertEqual(iadcommand.validator_01("0.5"), 0.5)

    def test_validator_01_invalid(self):
        """Invalid test for 11."""
        with self.assertRaises(argparse.ArgumentTypeError):
            iadcommand.validator_01("-1")

    def test_validator_11_valid(self):
        """Valid test that 11."""
        self.assertEqual(iadcommand.validator_11("-0.5"), -0.5)

    def test_validator_11_invalid(self):
        """Invalid test for 11."""
        with self.assertRaises(argparse.ArgumentTypeError):
            iadcommand.validator_11("2")

    def test_validator_positive_valid(self):
        """Valid test for positive."""
        self.assertEqual(iadcommand.validator_positive("10"), 10.0)

    def test_validator_positive_invalid(self):
        """Invalid test for positive."""
        with self.assertRaises(argparse.ArgumentTypeError):
            iadcommand.validator_positive("-5")


class TestIadFile(unittest.TestCase):
    """Tests for rxt files."""

    def test_valid_arguments(self):
        """Simple test."""
        test_args = ['iadcommand.py', 'data/basic-A.rxt']
        with self.assertRaises(SystemExit):
            with patch('sys.argv', test_args):
                iadcommand.main()


class TestIadCommandForward(unittest.TestCase):
    """Test forward calculation scenarios."""

    def test_only_albedo(self):
        """Test with only albedo provided."""
        test_args = ['iadcommand.py', '-z', '-a', '0.5']
        with self.assertRaises(SystemExit):
            with patch('sys.argv', test_args):
                iadcommand.main()

    def test_only_optical_thickness(self):
        """Test with only optical thickness provided."""
        test_args = ['iadcommand.py', '-z', '-b', '1']
        with self.assertRaises(SystemExit):
            with patch('sys.argv', test_args):
                iadcommand.main()

    def test_albedo_and_optical_thickness(self):
        """Test with albedo and optical thickness."""
        test_args = ['iadcommand.py', '-z', '-a', '0.1', '-b', '1']
        with patch('sys.argv', test_args):
            with self.assertRaises(SystemExit):  # Expecting the program to exit
                iadcommand.main()

    def test_albedo_and_optical_thickness_and_g(self):
        """Test with albedo and optical thickness and g."""
        test_args = ['iadcommand.py', '-z', '-a', '0.1', '-b', '1', '-g', '0.9']
        with self.assertRaises(SystemExit):
            with patch('sys.argv', test_args):
                iadcommand.main()

    def test_slab_index(self):
        """Test with slab index."""
        test_args = ['iadcommand.py', '-z', '-a', '0.1', '-n', '1.4']
        with self.assertRaises(SystemExit):
            with patch('sys.argv', test_args):
                iadcommand.main()

    def test_slab_and_slide_index(self):
        """Test with slab and slide index."""
        test_args = ['iadcommand.py', '-z', '-a', '0.1', '-n', '1.4', '-n', '1.5']
        with self.assertRaises(SystemExit):
            with patch('sys.argv', test_args):
                iadcommand.main()

    def test_only_absorption_coefficient(self):
        """Test with only absorption coefficient provided."""
        test_args = ['iadcommand.py', '-z', '--mua', '0.1']
        with self.assertRaises(SystemExit):
            with patch('sys.argv', test_args):
                iadcommand.main()

    def test_only_scattering_coefficient(self):
        """Test with only scattering coefficient provided."""
        test_args = ['iadcommand.py', '-z', '--mus', '10']
        with self.assertRaises(SystemExit):
            with patch('sys.argv', test_args):
                iadcommand.main()

    def test_only_anisotropy(self):
        """Test with only anisotropy provided."""
        test_args = ['iadcommand.py', '-z', '-g', '0.9']
        with self.assertRaises(SystemExit):
            with patch('sys.argv', test_args):
                iadcommand.main()

    def test_scattering_and_absorption(self):
        """Test with scattering and absorption."""
        test_args = ['iadcommand.py', '-z', '--mua', '0.1', '--mus', '10']
        with self.assertRaises(SystemExit):
            with patch('sys.argv', test_args):
                iadcommand.main()

    def test_scattering_and_absorption_and_g(self):
        """Test with scattering and absorption and g."""
        test_args = ['iadcommand.py', '-z', '--mua', '0.1', '--mus', '10', '-g', '0.9']
        with self.assertRaises(SystemExit):
            with patch('sys.argv', test_args):
                iadcommand.main()

    def test_invalid_albedo(self):
        """Test with invalid albedo value."""
        test_args = ['iadcommand.py', '-z', '-a', '-1']
        with patch('sys.argv', test_args):
            with self.assertRaises(SystemExit):
                iadcommand.main()

    def test_invalid_optical_thickness(self):
        """Test example with invalid argument values."""
        test_args = ['iadcommand.py', '-z', '-a', '0.5', '-b', '-1']
        with patch('sys.argv', test_args):
            with self.assertRaises(SystemExit):  # Expecting the program to exit
                iadcommand.main()

    def test_forward_calculation_missing_albedo(self):
        """Test example when albedo is not provided."""
        test_args = ['iadcommand.py', '-z', '-b', '1', '-g', '0.9']
        with patch('sys.argv', test_args):
            with self.assertRaises(SystemExit):  # Expecting the program to exit
                iadcommand.main()

    def test_invalid_scattering(self):
        """Test example with invalid argument values."""
        test_args = ['iadcommand.py', '-z', '--mus', '-0.5']
        with patch('sys.argv', test_args):
            with self.assertRaises(SystemExit):  # Expecting the program to exit
                iadcommand.main()

    def test_invalid_absorption(self):
        """Test example with invalid argument values."""
        test_args = ['iadcommand.py', '-z', '--mua', '-0.5']
        with patch('sys.argv', test_args):
            with self.assertRaises(SystemExit):  # Expecting the program to exit
                iadcommand.main()

    def test_invalid_anisotropy(self):
        """Test example with invalid argument values."""
        test_args = ['iadcommand.py', '-z', '-g', '1.1']
        with patch('sys.argv', test_args):
            with self.assertRaises(SystemExit):  # Expecting the program to exit
                iadcommand.main()


class TestIadSingle(unittest.TestCase):
    """Single sphere tests."""

    def test_inverse_missing(self):
        """Test with no arguments."""
        test_args = ['iadcommand.py']
        with patch('sys.argv', test_args):
            with self.assertRaises(SystemExit):
                iadcommand.main()

    def test_inverse_reflection(self):
        """Test with invalid albedo value."""
        test_args = ['iadcommand.py', '-r', '0.5']
        with patch('sys.argv', test_args):
            with self.assertRaises(SystemExit):
                iadcommand.main()

    def test_inverse_transmission(self):
        """Test with invalid albedo value."""
        test_args = ['iadcommand.py', '-t', '0.5']
        with patch('sys.argv', test_args):
            with self.assertRaises(SystemExit):
                iadcommand.main()

    def test_inverse_unscattered(self):
        """Test with invalid albedo value."""
        test_args = ['iadcommand.py', '-u', '0.5']
        with patch('sys.argv', test_args):
            with self.assertRaises(SystemExit):
                iadcommand.main()


if __name__ == '__main__':
    unittest.main()
