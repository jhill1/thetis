from thetis import *
import time as time_mod
from model_config import *

# Setup solver
mesh2d = read_mesh_from_checkpoint('bathymetry.h5')
solver_obj, start_time, update_forcings = construct_solver(
    mesh2d,
    output_directory="outputs_spinup",
    spinup=True,
    start_date=datetime.datetime(2022, 1, 1, tzinfo=sim_tz),
    end_date=datetime.datetime(2022, 1, 15, tzinfo=sim_tz),
    fields_to_export=["elev_2d", "uv_2d"],
    fields_to_export_hdf5=["elev_2d", "uv_2d"],
    simulation_export_time=1800.0,
)
solver_obj.assign_initial_conditions(uv=Constant((1.0e-10,1e-10, 1e-10)))
update_forcings(0.0)

# Time integrate
tic = time_mod.perf_counter()
solver_obj.iterate(update_forcings=update_forcings)
toc = time_mod.perf_counter()
print_output(f"Total duration: {toc-tic:.2f} seconds")
