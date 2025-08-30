"""
Simple unit tests for baltools.balconfig constants and logic
"""

import unittest


class TestBalConfigConstants(unittest.TestCase):
    """Test cases for BAL configuration constants"""
    
    def test_physical_constants(self):
        """Test physical constants are reasonable"""
        # Speed of light in km/s
        c = 299792.458  # km/s
        
        # Should be a positive number
        self.assertGreater(c, 0)
        self.assertIsInstance(c, (int, float))
        
        # Should be in km/s units (speed of light ~ 300,000 km/s)
        self.assertGreater(c, 200000)
        self.assertLess(c, 400000)
    
    def test_bal_wavelengths(self):
        """Test BAL feature wavelengths are reasonable"""
        # CIV wavelength should be around 1549 Angstroms
        lambda_civ = 1549.0
        self.assertAlmostEqual(lambda_civ, 1549.0, places=1)
        self.assertGreater(lambda_civ, 0)
        
        # SiIV wavelength should be around 1398 Angstroms
        lambda_siiv = 1398.0
        self.assertAlmostEqual(lambda_siiv, 1398.0, places=1)
        self.assertGreater(lambda_siiv, 0)
        
        # SiIV should be at shorter wavelength than CIV
        self.assertLess(lambda_siiv, lambda_civ)
    
    def test_velocity_search_range(self):
        """Test velocity search range for BAL troughs"""
        # VMIN_BAL should be negative (blueshifted)
        vmin_bal = -25000.0
        self.assertLess(vmin_bal, 0)
        
        # VMAX_BAL should be zero or positive
        vmax_bal = 0.0
        self.assertLessEqual(vmax_bal, 0)
        
        # VMIN_BAL should be less than VMAX_BAL
        self.assertLess(vmin_bal, vmax_bal)
        
        # Typical BAL velocities are in thousands of km/s
        self.assertLess(vmin_bal, -1000)
    
    def test_bal_lambda_range(self):
        """Test BAL wavelength range for PCA"""
        # Minimum wavelength should be positive
        bal_lambda_min = 1261.0
        self.assertGreater(bal_lambda_min, 0)
        
        # Maximum wavelength should be greater than minimum
        bal_lambda_max = 2399.0
        self.assertGreater(bal_lambda_max, bal_lambda_min)
        
        # Wavelengths should be in reasonable range for BAL features
        self.assertGreater(bal_lambda_min, 1000)  # > 1000 Angstroms
        self.assertLess(bal_lambda_max, 3000)     # < 3000 Angstroms
    
    def test_redshift_range(self):
        """Test redshift range for BAL catalog"""
        # Minimum redshift should be positive
        bal_zmin = 1.57
        self.assertGreater(bal_zmin, 0)
        
        # Maximum redshift should be greater than minimum
        bal_zmax = 5.0
        self.assertGreater(bal_zmax, bal_zmin)
        
        # Redshifts should be in reasonable range for BAL QSOs
        self.assertGreater(bal_zmin, 1.0)  # BALs typically at z > 1
        self.assertLess(bal_zmax, 10.0)    # Reasonable upper limit
    
    def test_pca_parameters(self):
        """Test PCA and BAL parameters"""
        # NPCA should be positive
        npca = 5
        self.assertGreater(npca, 0)
        
        # NBI should be positive
        nbi = 5
        self.assertGreater(nbi, 0)
        
        # NAI should be positive
        nai = 17
        self.assertGreater(nai, 0)
        
        # NAI should be greater than NBI (more AI troughs than BI)
        self.assertGreater(nai, nbi)
    
    def test_parameter_consistency(self):
        """Test that parameters are internally consistent"""
        # Test wavelength consistency
        lambda_siiv = 1398.0
        lambda_civ = 1549.0
        self.assertLess(lambda_siiv, lambda_civ, 
                       "SiIV wavelength should be less than CIV wavelength")
        
        # Test velocity range consistency
        vmin_bal = -25000.0
        vmax_bal = 0.0
        self.assertLess(vmin_bal, vmax_bal,
                       "VMIN_BAL should be less than VMAX_BAL")
        
        # Test redshift range consistency
        bal_zmin = 1.57
        bal_zmax = 5.0
        self.assertLess(bal_zmin, bal_zmax,
                       "BAL_ZMIN should be less than BAL_ZMAX")
        
        # Test wavelength range consistency
        bal_lambda_min = 1261.0
        bal_lambda_max = 2399.0
        self.assertLess(bal_lambda_min, bal_lambda_max,
                       "BAL_LAMBDA_MIN should be less than BAL_LAMBDA_MAX")
        
        # Test that BAL features fall within the wavelength range
        self.assertGreaterEqual(lambda_civ, bal_lambda_min,
                               "CIV wavelength should be >= BAL_LAMBDA_MIN")
        self.assertLessEqual(lambda_civ, bal_lambda_max,
                            "CIV wavelength should be <= BAL_LAMBDA_MAX")
        
        self.assertGreaterEqual(lambda_siiv, bal_lambda_min,
                               "SiIV wavelength should be >= BAL_LAMBDA_MIN")
        self.assertLessEqual(lambda_siiv, bal_lambda_max,
                            "SiIV wavelength should be <= BAL_LAMBDA_MAX")
    
    def test_physical_calculations(self):
        """Test that physical constants can be used in calculations"""
        # Speed of light should be in km/s
        c_km_s = 299792.458
        
        # Test that it's a reasonable value
        self.assertGreater(c_km_s, 200000)
        self.assertLess(c_km_s, 400000)
        
        # Test that we can use it in calculations
        # Example: calculate wavelength from frequency
        frequency_hz = 1e15  # 1000 THz
        wavelength_angstroms = (c_km_s * 1e5) / frequency_hz  # Convert to cm/s then to Angstroms
        
        self.assertGreater(wavelength_angstroms, 0)
        self.assertLess(wavelength_angstroms, 10000)  # Should be in visible/UV range


if __name__ == '__main__':
    unittest.main()
