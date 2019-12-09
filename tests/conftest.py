# -*- coding: utf-8 -*-

"""
Configuration for tests.
"""

from hypothesis import settings

settings.register_profile("dit", deadline=None)
settings.load_profile("dit")
