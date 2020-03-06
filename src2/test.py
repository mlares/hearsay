#import sys
#sys.path.insert(1, './')

import pytest


class testClass:

    def inc(x):
        return x + 1

    def test_answer():
        assert inc(3) == 4





# class test_CCN:
#     """
#     This module is intended to store test functions
#     related to CCNs and communication networks.
#     """
# 
# #    def test_node_single_contact(self):
# #        #{{{
# #        """
# #        test_node_single_contact(self):
# #            test the assignment of a songle contact.
# #
# #        Tasks:
# #
# #        Args:
# #
# #        Raises:
# #            errors?
# #
# #        Returns:
# #        """
# 
#     def test_answer():
#         assert inc(3) == 5
#      
