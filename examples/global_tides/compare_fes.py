from thetis import *
import hrds
import numpy as np
from model_config import *
import math

# 1) Load the mesh from bathymetry.h5 checkpoint file
with CheckpointFile("bathymetry.h5", "r") as chk_mesh:
    mesh2d = chk_mesh.load_mesh("mesh")

mesh2d.init_cell_orientations(SpatialCoordinate(mesh2d))

# Deal with FES2022 interpolation
with timed_stage('loading FES'):
    lon, lat = coords_xyz_to_lonlat(mesh2d)
    fes = hrds.HRDS("m2_fes2022.nc", global_data=True)
    fes.set_bands("amplitude")
    
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)
    fes_amp = Function(P1_2d, name="fes_amp")
    
    amp_vector = fes_amp.dat.data
    xvector = mesh2d.coordinates.dat.data
    
    assert xvector.shape[0] == amp_vector.shape[0]
    for i, xyz in enumerate(mesh2d.coordinates.dat.data):
        p_lon = math.degrees(lon.evaluate(xyz, None, None, None))
        p_lat = math.degrees(lat.evaluate(xyz, None, None, None))
        amp_vector[i] = fes.get_val((p_lon, p_lat))

# 2) Load in the model diagnostic file (analysis/M2_amp.h5)
with CheckpointFile("analysis/M2_amp.h5", "r") as chk_mod:
    model_amp = chk_mod.load_function(mesh2d, name="M2_amp")

# 3) Perform the difference
# Create a new function on the same space to hold the computed difference
amp_diff = Function(P1_2d, name="amplitude_difference")
amp_diff.assign(fes_amp - model_amp)

# Calculate the L2 norm of the difference function
l2_difference = norm(amp_diff, norm_type='L2')

print("\n" + "="*50)
print(f"L2 Norm of the difference: {l2_difference}")
print("="*50 + "\n")

# 4) Output the difference to a VTK file (.pvd)
# This will create a directory/file structure ready for ParaView
vtk_diff = VTKFile("analysis/M2_amp_difference.pvd")
vtk_diff.write(amp_diff)

# Save out the final FES dataset checkpoint
with CheckpointFile('fes_m2.h5', 'w') as chk:
    chk.save_mesh(mesh2d)
    chk.save_function(fes_amp, name='fes_amp')
