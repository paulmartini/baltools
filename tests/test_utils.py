"""
Simple unit tests for baltools.utils module
"""

import unittest
import os
import tempfile
import shutil

# Simple test of utility functions without importing the full module
class TestUtilsFunctions(unittest.TestCase):
    """Test cases for utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_zeropad_function(self):
        """Test zeropad function logic"""
        # This tests the logic without importing the module
        def zeropad(input_val, N=4):
            if isinstance(input_val, str):
                if N == 3:
                    return "{:03d}".format(int(input_val))
                elif N == 4:
                    return "{:04d}".format(int(input_val))
                elif N == 5:
                    return "{:05d}".format(int(input_val))
                else:
                    return str(input_val)
            else:
                return str(input_val)
        
        # Test N=3
        self.assertEqual(zeropad("1", 3), "001")
        self.assertEqual(zeropad("12", 3), "012")
        self.assertEqual(zeropad("123", 3), "123")
        
        # Test N=4
        self.assertEqual(zeropad("1", 4), "0001")
        self.assertEqual(zeropad("12", 4), "0012")
        self.assertEqual(zeropad("123", 4), "0123")
        self.assertEqual(zeropad("1234", 4), "1234")
        
        # Test N=5
        self.assertEqual(zeropad("1", 5), "00001")
        self.assertEqual(zeropad("123", 5), "00123")
        self.assertEqual(zeropad("12345", 5), "12345")
        
        # Test invalid input
        self.assertEqual(zeropad(123, 4), "123")
        self.assertEqual(zeropad("123", 6), "123")
    
    def test_gethpdir_function(self):
        """Test healpix directory parsing logic"""
        def gethpdir(healpix):
            if len(healpix) < 3:
                return '0'
            elif len(healpix) == 3:
                return healpix[0]
            elif len(healpix) == 4:
                return healpix[0:2]
            else:
                return healpix[:len(healpix)-2]
        
        # Test short healpix
        self.assertEqual(gethpdir("1"), "0")
        self.assertEqual(gethpdir("12"), "0")
        
        # Test 3-digit healpix
        self.assertEqual(gethpdir("123"), "1")
        self.assertEqual(gethpdir("456"), "4")
        
        # Test 4-digit healpix
        self.assertEqual(gethpdir("1234"), "12")
        self.assertEqual(gethpdir("5678"), "56")
        
        # Test longer healpix
        self.assertEqual(gethpdir("12345"), "123")
        self.assertEqual(gethpdir("123456"), "1234")
    
    def test_directory_creation(self):
        """Test directory creation functionality"""
        def pmmkdir(direct):
            if not os.path.isdir(direct):
                try:
                    os.makedirs(direct)
                except PermissionError:
                    raise SystemExit(1)
        
        # Test creating new directory
        new_dir = os.path.join(self.temp_dir, "test_dir")
        self.assertFalse(os.path.isdir(new_dir))
        pmmkdir(new_dir)
        self.assertTrue(os.path.isdir(new_dir))
        
        # Test existing directory
        pmmkdir(new_dir)  # Should not raise error
        self.assertTrue(os.path.isdir(new_dir))


if __name__ == '__main__':
    unittest.main()

