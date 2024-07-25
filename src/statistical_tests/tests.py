# GDST, CCT, NPT, cluster_tendency_test, reconstruction_test
# TODO: what is this one?
from statistical_tests.CCT import main as cct


from statistical_tests.CTT import ctt
from statistical_tests.GDST import main as gdst
from statistical_tests.NPT import main as npt
from statistical_tests.RCT import main as rct


# TODO: all tests do not have a deterministic random state.
# consider adding this for reproducibility.
