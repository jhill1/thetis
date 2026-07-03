from thetis import *
import hrds
import numpy as np
from model_config import *
import math

mesh2d = Mesh("world_40km.msh", dim=3) # mesh file
mesh2d.init_cell_orientations(SpatialCoordinate(mesh2d))

def smoothen_bathymetry(bathymetry2d): # smoothing bathymetry
    v = TestFunction(bathymetry2d.function_space())
    massb = assemble(v * bathymetry2d *dx)
    massl = assemble(v*dx)
    with massl.dat.vec as ml, massb.dat.vec as mb, bathymetry2d.dat.vec as sb:
        ml.reciprocal()
        sb.pointwiseMult(ml, mb)

# first deal with bathymetry
with timed_stage('initialising bathymetry'):
    lon,lat = coords_xyz_to_lonlat(mesh2d)
    bathy = hrds.HRDS("GEBCO_2026_0.1.nc", global_data=True)
    bathy.set_bands()
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry2d = Function(P1_2d, name="bathymetry")
    bvector = bathymetry2d.dat.data
    xvector = mesh2d.coordinates.dat.data
    assert xvector.shape[0]==bvector.shape[0]
    for i, (xyz) in enumerate(mesh2d.coordinates.dat.data):
        p_lon = math.degrees(lon.evaluate(xyz, None, None, None))
        p_lat = math.degrees(lat.evaluate(xyz, None, None, None))
        bvector[i] = -1.0 * bathy.get_val((p_lon, p_lat))

    bvector[bvector < 50] = 50

smoothen_bathymetry(bathymetry2d)
chk = CheckpointFile('bathymetry.h5', 'w')
chk.save_mesh(mesh2d)
chk.save_function(bathymetry2d, name='bathymetry')
vtk_file = VTKFile("bathy.pvd")
vtk_file.write(bathymetry2d)
