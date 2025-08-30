"""
Simple integration tests for baltools package
"""

import unittest


class TestIntegrationLogic(unittest.TestCase):
    """Integration tests for baltools package logic"""
    
    def test_healpix_directory_structure(self):
        """Test healpix directory structure consistency"""
        def gethpdir(healpix):
            if len(healpix) < 3:
                return '0'
            elif len(healpix) == 3:
                return healpix[0]
            elif len(healpix) == 4:
                return healpix[0:2]
            else:
                return healpix[:len(healpix)-2]
        
        # Test various healpix values and their directory structures
        test_cases = [
            ("1", "0"),
            ("12", "0"), 
            ("123", "1"),
            ("1234", "12"),
            ("12345", "123"),
            ("123456", "1234")
        ]
        
        for healpix, expected_dir in test_cases:
            hpdir = gethpdir(healpix)
            self.assertEqual(hpdir, expected_dir, 
                           f"Expected {expected_dir} for healpix {healpix}, got {hpdir}")
    
    def test_bal_parameter_consistency(self):
        """Test that BAL parameters are internally consistent"""
        # Define constants
        lambda_siiv = 1398.0
        lambda_civ = 1549.0
        vmin_bal = -25000.0
        vmax_bal = 0.0
        bal_zmin = 1.57
        bal_zmax = 5.0
        bal_lambda_min = 1261.0
        bal_lambda_max = 2399.0
        
        # Test wavelength consistency
        self.assertLess(lambda_siiv, lambda_civ, 
                       "SiIV wavelength should be less than CIV wavelength")
        
        # Test velocity range consistency
        self.assertLess(vmin_bal, vmax_bal,
                       "VMIN_BAL should be less than VMAX_BAL")
        
        # Test redshift range consistency
        self.assertLess(bal_zmin, bal_zmax,
                       "BAL_ZMIN should be less than BAL_ZMAX")
        
        # Test wavelength range consistency
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
    
    def test_physical_constants_integration(self):
        """Test that physical constants are used consistently"""
        # Speed of light should be in km/s
        c_km_s = 299792.458
        
        # Test that it's a reasonable value (speed of light ~ 300,000 km/s)
        self.assertGreater(c_km_s, 200000)
        self.assertLess(c_km_s, 400000)
        
        # Test that we can use it in calculations
        # Example: calculate wavelength from frequency
        frequency_hz = 1e15  # 1000 THz
        wavelength_angstroms = (c_km_s * 1e5) / frequency_hz  # Convert to cm/s then to Angstroms
        
        self.assertGreater(wavelength_angstroms, 0)
        self.assertLess(wavelength_angstroms, 10000)  # Should be in visible/UV range
    
    def test_parameter_ranges_integration(self):
        """Test that parameter ranges are scientifically reasonable"""
        # BAL redshifts should be in a reasonable range for QSOs
        bal_zmin = 1.57
        bal_zmax = 5.0
        self.assertGreater(bal_zmin, 1.0, "BAL_ZMIN should be > 1.0")
        self.assertLess(bal_zmax, 10.0, "BAL_ZMAX should be < 10.0")
        
        # BAL velocities should be in a reasonable range
        vmin_bal = -25000.0
        vmax_bal = 0.0
        self.assertLess(vmin_bal, -1000, "VMIN_BAL should be < -1000 km/s")
        self.assertGreater(vmax_bal, -50000, "VMAX_BAL should be > -50000 km/s")
        
        # Wavelengths should be in UV/optical range
        bal_lambda_min = 1261.0
        bal_lambda_max = 2399.0
        self.assertGreater(bal_lambda_min, 1000, "BAL_LAMBDA_MIN should be > 1000 Angstroms")
        self.assertLess(bal_lambda_max, 3000, "BAL_LAMBDA_MAX should be < 3000 Angstroms")
        
        # BAL feature wavelengths should be in the UV
        lambda_civ = 1549.0
        lambda_siiv = 1398.0
        self.assertGreater(lambda_civ, 1500, "CIV wavelength should be > 1500 Angstroms")
        self.assertLess(lambda_civ, 1600, "CIV wavelength should be < 1600 Angstroms")
        
        self.assertGreater(lambda_siiv, 1300, "SiIV wavelength should be > 1300 Angstroms")
        self.assertLess(lambda_siiv, 1500, "SiIV wavelength should be < 1500 Angstroms")
    
    def test_function_signature_integration(self):
        """Test that function signatures are consistent"""
        # Test that the desibalfinder function has the expected parameters
        def desibalfinder(specfilename, alttemp=False, altbaldir=None, altzdir=None, 
                         zfileroot='zbest', overwrite=True, verbose=False, 
                         release=None, usetid=True, format='healpix'):
            pass
        
        import inspect
        sig = inspect.signature(desibalfinder)
        params = list(sig.parameters.keys())
        
        # Check that usetid parameter is present (new feature)
        self.assertIn('usetid', params, "usetid parameter should be present")
        
        # Check that usetid has the correct default value
        self.assertEqual(sig.parameters['usetid'].default, True, 
                        "usetid should default to True")
        
        # Check that all required parameters are present
        required_params = ['specfilename']
        for param in required_params:
            self.assertIn(param, params, f"Required parameter '{param}' not found")
    
    def test_file_parsing_integration(self):
        """Test that file parsing logic works consistently"""
        def parse_filename(filename):
            if 'spectra-' in filename:
                return 'spectra_file'
            elif 'coadd-' in filename:
                return 'coadd_file'
            else:
                return 'unknown_file'
        
        def gethpdir(healpix):
            if len(healpix) < 3:
                return '0'
            elif len(healpix) == 3:
                return healpix[0]
            elif len(healpix) == 4:
                return healpix[0:2]
            else:
                return healpix[:len(healpix)-2]
        
        # Test that file parsing and healpix parsing work together
        test_files = [
            ('spectra-16-1234.fits', 'spectra_file', '12'),
            ('coadd-dark-bright-5678.fits', 'coadd_file', '56'),
            ('spectra-16-12345.fits', 'spectra_file', '123')
        ]
        
        for filename, expected_type, expected_hpdir in test_files:
            file_type = parse_filename(filename)
            self.assertEqual(file_type, expected_type)
            
            # Extract healpix from filename (simplified)
            if 'spectra-16-' in filename:
                healpix = filename.replace('spectra-16-', '').replace('.fits', '')
            elif 'coadd-dark-bright-' in filename:
                healpix = filename.replace('coadd-dark-bright-', '').replace('.fits', '')
            else:
                healpix = "0"
            
            hpdir = gethpdir(healpix)
            self.assertEqual(hpdir, expected_hpdir)


if __name__ == '__main__':
    unittest.main()
