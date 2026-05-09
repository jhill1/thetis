import pytest
import numpy as np
from firedrake import *
from tidal_module import TidalForcing # Assuming your class is in tidal_module.py

@pytest.fixture
def mesh():
    # Use a coarse UnitSphereMesh for testing global logic
    # Refinement level 3 is enough to get non-zero gradients
    return UnitSphereMesh(3)

@pytest.fixture
def tidal(mesh):
    return TidalForcing(mesh, l_smooth=100000.0)

def test_astronomical_arguments_length(tidal):
    """Verify find_chi returns the correct number of constituents."""
    chi = tidal.find_chi(0.0)
    assert len(chi) == 11
    assert isinstance(chi, np.ndarray)

def test_chi_time_evolution(tidal):
    """Verify that chi changes over a 12-hour period."""
    chi_t0 = tidal.find_chi(0.0)
    chi_t12h = tidal.find_chi(12 * 3600.0)
    # They should not be identical
    assert not np.allclose(chi_t0, chi_t12h)

def test_sal_linearity(mesh, tidal):
    """The SAL PDE is linear; doubling eta should double the SAL potential."""
    V = tidal.sal_potential.function_space()
    eta = Function(V)
    
    # Case 1: Constant unit elevation
    eta.assign(1.0)
    forcing_1 = Function(V)
    tidal.update_forcing(forcing_1, eta, 0.0)
    sal_1 = tidal.sal_potential.copy(deepcopy=True)
    
    # Case 2: Double elevation
    eta.assign(2.0)
    forcing_2 = Function(V)
    tidal.update_forcing(forcing_2, eta, 0.0)
    sal_2 = tidal.sal_potential.copy(deepcopy=True)
    
    # Check linearity: sal_2 should be approx 2 * sal_1
    # We use a small tolerance for solver noise
    val1 = sal_1.dat.data_ro.max()
    val2 = sal_2.dat.data_ro.max()
    assert np.isclose(val2, 2 * val1, rtol=1e-5)

def test_equilibrium_tide_magnitude(mesh, tidal):
    """Verify that the forcing field produces physically sensible values (~0.1-0.5m)."""
    V = tidal.sal_potential.function_space()
    eta = Function(V).assign(0.0) # No SAL effect for this test
    forcing = Function(V)
    
    # Update at t=0
    tidal.update_forcing(forcing, eta, 0.0)
    
    max_val = forcing.dat.data_ro.max()
    min_val = forcing.dat.data_ro.min()
    
    # Global equilibrium tides are usually within +/- 0.6 meters
    assert max_val < 1.0
    assert min_val > -1.0
    # Ensure it's not just returning zeros
    assert not np.isclose(max_val, 0.0)

def test_sal_smoothing(mesh, tidal):
    """Verify that l_smooth correctly spreads a localized spike in eta."""
    V = tidal.sal_potential.function_space()
    eta = Function(V)
    
    # Create a delta-like spike at one node
    eta.dat.data[0] = 10.0 
    
    tidal.update_forcing(Function(V), eta, 0.0)
    sal = tidal.sal_potential
    
    # In a Helmholtz solve, a spike in input should lead to a 
    # smooth, distributed output.
    # The max of the output should be much lower than the max of the input spike.
    assert sal.dat.data_ro.max() < 1.0
