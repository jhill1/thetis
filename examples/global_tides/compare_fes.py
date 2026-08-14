from thetis import *
import hrds
import numpy as np
from model_config import *
import math
import netCDF4
import scipy.interpolate as si


def coords_xyz_to_lonlat(mesh):
    """
    Convert Earth-centered Cartesian coordinates to (longitude, latitude)
    """
    x, y, z = SpatialCoordinate(mesh)
    z_norm = z / sqrt(x**2 + y**2 + z**2)
    z_norm = min_value(max_value(z_norm, -1.0), 1.0)  # avoid silly roundoff errors
    lat = asin(z_norm)
    lon = atan2(y, x)
    return lon, lat


# Load in the model diagnostic file (analysis_visc1000/tidal_stats_scal.h5)
with CheckpointFile("analysis_visc100000/tidal_stats_scal.h5", "r") as chk_mod:
    mesh2d = chk_mod.load_mesh()
    model_amp_dg = chk_mod.load_function(mesh2d, name="M2_amp")

P1_2d = FunctionSpace(mesh2d, 'CG', 1)
# Project from the model's native DG space to your continuous CG space (P1_2d)
model_amp = project(model_amp_dg, P1_2d)

# first deal with loading FES in
with timed_stage('initialising fes'):
    lon,lat = coords_xyz_to_lonlat(mesh2d)
    fes = hrds.HRDS("fes_m2_amp.tif", global_data=True)
    fes.set_bands()
    fes2d = Function(P1_2d, name="fes_m2")
    fes_vector = fes2d.dat.data
    xvector = mesh2d.coordinates.dat.data
    assert xvector.shape[0]==fes_vector.shape[0]
    for i, (xyz) in enumerate(mesh2d.coordinates.dat.data):
        p_lon = math.degrees(lon.evaluate(xyz, None, None, None))
        p_lat = math.degrees(lat.evaluate(xyz, None, None, None))
        fes_vector[i] = fes.get_val((p_lon, p_lat))

    fes_vector[fes_vector < 0] = 0
    fes_vector[fes_vector > 10000] = 0
    
# Perform the difference
# Create a new function on the same space to hold the computed difference
amp_diff = Function(P1_2d, name="amplitude_difference")
amp_diff.assign(model_amp-(fes2d/100.0))

# Calculate the L2 norm of the difference function
l2_difference = norm(amp_diff, norm_type='L2')

print("\n" + "="*50)
print(f"L2 Norm of the difference: {l2_difference}")
print("="*50 + "\n")

# 4) Output the difference to a VTK file (.pvd)
# This will create a directory/file structure ready for ParaView
vtk_diff = VTKFile("analysis/M2_amp_difference.pvd")
vtk_diff.write(amp_diff)

fes_vtk = VTKFile("fes_m2.pvd")
fes_vtk.write(fes2d)
# Save out the final FES dataset checkpoint
with CheckpointFile('fes_m2.h5', 'w') as chk:
    chk.save_mesh(mesh2d)
    chk.save_function(fes2d, name='fes_amp')
