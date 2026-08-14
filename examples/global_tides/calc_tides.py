import numpy as np
import uptide
from thetis import *
import sys
import os.path
from firedrake.petsc import PETSc
import gc
from model_config import *

# Where should the output of this analysis go
output_dir = 'analysis'
create_directory(output_dir)

thetis_dir = "outputs_spinup"
t_end = 3888000 
t_export = 3600
t_start = 1296000

# 1. Load the mesh from the first checkpoint
# (Firedrake automatically handles parallel distribution if run with mpiexec)
with CheckpointFile(os.path.join(thetis_dir, "hdf5", "Elevation2d_00000.h5"), 'r') as chk:
    thetis_mesh = chk.load_mesh()

# Calculate temporal steps
t_n = int((t_end - t_start) / t_export) + 1
thetis_times = t_start + t_export * np.arange(t_n)

# Define our scalar Function Space
P1DG = FunctionSpace(thetis_mesh, "DG", 1)
elev = Function(P1DG, name='elev_2d')

# Local number of nodes on THIS MPI core
local_nodes = elev.dat.data.shape[0]

# Allocate local memory block (Each core only allocates its own chunk of the global mesh!)
elev_data_set = np.empty((t_n, local_nodes), dtype=np.single)

# 2. Parallel I/O: Loop through checkpoints and load local partition data
count = 0
for t in thetis_times:
    iexport = int(t / t_export)
    filename = '{0:s}_{1:05d}'.format("Elevation2d", iexport)
    
    # Only print on root node to prevent terminal clutter in parallel
    print_output(f"Loading export step: {filename}")
    
    with CheckpointFile(os.path.join(thetis_dir, "hdf5", filename + ".h5"), 'r') as afile:
        e = afile.load_function(thetis_mesh, "elev_2d")
        elev_data_set[count, :] = e.dat.data[:]
        
        # Free memory aggressively inside the loop
        PETSc.garbage_cleanup(comm=afile.comm)
        gc.collect()
    count += 1

print_output("Working out min/max/range")
max_fs = np.max(elev_data_set, axis=0)
min_fs = np.min(elev_data_set, axis=0)
detector_tidal_range = max_fs - min_fs

print_output("Configuring tidal constituents...")
constituents = ["M2", "S2", "O1", "K1"]

tide = uptide.Tides(constituents)
tide.set_initial_time(datetime.datetime(2022, 1, 1, tzinfo=sim_tz))

# Pre-allocate containers for harmonic outputs
num_constituents = len(constituents)
detector_amplitudes = np.empty((local_nodes, num_constituents))
detector_phases = np.empty((local_nodes, num_constituents))

# OPTIMIZATION: Demean the entire data set simultaneously along the time axis
elev_demeaned = elev_data_set - np.mean(elev_data_set, axis=0)

print_output("Performing parallel harmonic analysis...")
for i in range(local_nodes):
    # Perform harmonic fit on a single node's timeline
    thetis_amplitudes, thetis_phases = uptide.analysis.harmonic_analysis(
        tide, elev_demeaned[:, i], thetis_times
    )
    detector_amplitudes[i, :] = thetis_amplitudes
    detector_phases[i, :] = thetis_phases

print_output("Analysis complete. Writing parallel Checkpoint HDF5 and PVD outputs...")

with CheckpointFile(os.path.join(output_dir, 'tidal_stats_scal.h5'), "w") as chk:
    chk.save_mesh(thetis_mesh)
    
    tr = Function(P1DG, name="TidalRange")
    tr.dat.data[:] = detector_tidal_range
    chk.save_function(tr)
    VTKFile(os.path.join(output_dir, 'tidal_range.pvd')).write(tr)

    for idx, const_name in enumerate(constituents):
        amp = Function(P1DG, name=f"{const_name}_amp")
        phase = Function(P1DG, name=f"{const_name}_phase")
        phasepi = Function(P1DG, name=f"{const_name}_phasepi")
        
        amp.dat.data[:] = detector_amplitudes[:, idx]
        phase.dat.data[:] = detector_phases[:, idx]
        phasepi.dat.data[:] = np.arcsin(np.sin(phase.dat.data[:]))
        
        chk.save_function(amp)
        chk.save_function(phase)
        chk.save_function(phasepi)
        
        VTKFile(os.path.join(output_dir, f'{const_name}_amp.pvd')).write(amp)
        VTKFile(os.path.join(output_dir, f'{const_name}_phase.pvd')).write(phase)
        VTKFile(os.path.join(output_dir, f'{const_name}_phase_mod_pi.pvd')).write(phasepi)

with CheckpointFile(os.path.join(output_dir, 'temporal_stats_elev.h5'), "w") as chk:
    chk.save_mesh(thetis_mesh)
    
    maxfs = Function(P1DG, name="MaxFS")
    maxfs.dat.data[:] = max_fs
    chk.save_function(maxfs)
    VTKFile(os.path.join(output_dir, 'max_fs.pvd')).write(maxfs)
    
    minfs = Function(P1DG, name="MinFS")
    minfs.dat.data[:] = min_fs
    chk.save_function(minfs)
    VTKFile(os.path.join(output_dir, 'min_fs.pvd')).write(minfs)

print_output("All tasks successfully completed!")
