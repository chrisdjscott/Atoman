
"""
Unit tests for filters.base module

"""
import unittest

import numpy as np

from .. import base

class TestBaseSettings(unittest.TestCase):
    """
    Test filters.base.BaseSettings
    
    """
    def test_registerSetting(self):
        """
        Filter settings register setting
        
        """
        settings = base.BaseSettings()
        
        # add new setting with default value (None)
        settings.registerSetting("myTestSetting")
        self.assertIsNone(settings._settings["myTestSetting"])
        
        # add new setting with specified setting
        settings.registerSetting("anotherSetting", 12.0)
        self.assertEqual(settings._settings["anotherSetting"], 12.0)
        
        # register array
        settings.registerSetting("anArray", [1, 4, 6])
        self.assertEqual(settings._settings["anArray"][0], 1)
        self.assertEqual(settings._settings["anArray"][1], 4)
        self.assertEqual(settings._settings["anArray"][2], 6)
        
        # check raises error if already exists
        with self.assertRaises(ValueError):
            settings.registerSetting("myTestSetting")
    
    def test_updateSetting(self):
        """
        Filter settings update setting
        
        """
        settings = base.BaseSettings()
        
        # add a couple of settings manually
        settings._settings["firstSetting"] = 1
        settings._settings["secondSetting"] = "hello"
        settings._settings["arraySetting"] = [1,2,3]
        
        # modify a setting
        settings.updateSetting("firstSetting", 33)
        self.assertEqual(settings._settings["firstSetting"], 33)
        
        # modify array setting
        settings.updateSettingArray("arraySetting", 0, 5)
        self.assertEqual(settings._settings["arraySetting"][0], 5)
        
        # bad index
        with self.assertRaises(IndexError):
            settings.updateSettingArray("arraySetting", 4, 9)
        
        # update scalar using array method
        with self.assertRaises(TypeError):
            settings.updateSettingArray("firstSetting", 0, 19)
        
        # update array using scalar method
        with self.assertRaises(TypeError):
            settings.updateSetting("arraySetting", 22)
    
    def test_getSetting(self):
        """
        Filter settings get setting
        
        """
        settings = base.BaseSettings()
        
        # add a couple of settings manually
        settings._settings["firstSetting"] = 1
        settings._settings["secondSetting"] = "hello"
        settings._settings["arraySetting"] = [1,2,3]
        
        # get scalar setting
        val = settings.getSetting("firstSetting")
        self.assertEqual(val, 1)
        
        # get array setting
        val = settings.getSetting("arraySetting")
        self.assertEqual(val[0], 1)
        self.assertEqual(val[1], 2)
        self.assertEqual(val[2], 3)
        
        # exception if not exist
        with self.assertRaises(ValueError):
            settings.getSetting("badSetting")
