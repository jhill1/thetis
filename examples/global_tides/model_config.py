from thetis import *
import thetis.forcing as forcing
import os
import numpy
import types
import csv

# Setup zones
sim_tz = timezone.pytz.utc

r_earth = 6371220.  # radius of Earth
omega = 7.292e-5  # Earth's angular velocity

def coords_lonlat_to_xyz(lon, lat):
    """
    Convert (longitude, latitude) on a spherical shell of given radius 
    to Earth-centered Cartesian coordinates (x, y, z).
    """
    # x = R * cos(lat) * cos(lon)
    # y = R * cos(lat) * sin(lon)
    # z = R * sin(lat)
    
    x = r_earth * cos(lat) * cos(lon)
    y = r_earth * cos(lat) * sin(lon)
    z = r_earth * sin(lat)
    
    return x, y, z

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



def read_station_data():
    """
    Load tide gauge metadata from the CSV file
    `stations_elev.csv`.

    :return: a dictionary containing gauge locations
        in latitude-longitude coordinates and the
        corresponding region code used in the CMEMS
        database
    """
    pwd = "."#os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(pwd, "uhslc_station_locations.csv"), "r") as csvfile:
        stations = {
            d["Station Name"]: {
                "latlon": (float(d["Latitude"]), float(d["Longitude"])),
            }
            for d in csv.DictReader(csvfile, delimiter=",", skipinitialspace=True)
        }
    return stations


def construct_solver(mesh2d, spinup=False, store_station_time_series=True, **model_options):
    """
    Construct a :class:`FlowSolver2d` instance for inverse modelling
    in the North Sea.

    :kwarg mesh2d: 2D mesh to use
    :kwarg spinup: is this a spin-up run, or a subsequent simulation?
    :kwarg store_station_time_series: should gauge measurements be
        stored to disk?
    :return: :class:`FlowSolver2d` instance, the start date for the
        simulation and a function for updating forcings
    """

    pwd = os.path.abspath(os.path.dirname(__file__))
    h5_file_name = os.path.join(pwd, "bathymetry.h5")
    with CheckpointFile(h5_file_name, "r") as f:
        bathymetry_2d = f.load_function(mesh2d, "bathymetry")

    # Setup mesh and lonlat coords
    lon, lat =  coords_xyz_to_lonlat(mesh2d)

    # Setup Manning friction
    P1_2d = get_functionspace(mesh2d, "CG", 1)
    manning_2d = Function(P1_2d, name="Manning coefficient")
    manning_2d.assign(3.0e-02)

    # Setup Coriolis forcing
    x, y, z = SpatialCoordinate(mesh2d)
    f_expr = 2 * omega * z / r_earth
    coriolis_2d = Function(P1_2d)
    coriolis_2d.interpolate(f_expr)

    # Setup temporal discretisation
    default_start_date = datetime.datetime(2022, 1, 1, tzinfo=sim_tz)
    default_end_date = datetime.datetime(2022, 1, 2, tzinfo=sim_tz)
    start_date = model_options.pop("start_date", default_start_date)
    end_date = model_options.pop("end_date", default_end_date)
    dt = 180
    t_export = 3600.0
    t_end = (end_date - start_date).total_seconds()


    # Create solver
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options
    options.element_family = "bdm-dg"
    options.polynomial_degree = 1
    options.coriolis_frequency = coriolis_2d
    options.manning_drag_coefficient = manning_2d
    options.simulation_initial_date = start_date
    options.simulation_end_date = end_date
    options.simulation_export_time = t_export
    options.swe_timestepper_type = 'CrankNicolson'
    options.timestep = dt
    options.check_volume_conservation_2d = True
    options.fields_to_export = ["elev_2d", "uv_2d"]
    options.fields_to_export_hdf5 = ["elev_2d", "uv_2d"]
    options.horizontal_viscosity = Constant(100)
    options.use_wetting_and_drying = True
    #options.use_automatic_wetting_and_drying_alpha = True
    #options.wetting_and_drying_alpha_min = Constant(0.5)
    #options.wetting_and_drying_alpha_max = Constant(100.0)
    options.wetting_and_drying_alpha = Constant(100.0)
    options.update(model_options)

    tidal_forcing = forcing.EquilibriumTidalForcing(mesh2d, l_smooth=150000.0)
    total_forcing_meters = Function(P1_2d, name="Tidal_Forcing_Meters")
    p_atm_tidal = Function(P1_2d, name="Tidal_Pressure_Pascals")

    options.atmospheric_pressure = p_atm_tidal

    # Constant physical parameters
    rho_0 = 1025.0  # Reference water density (kg/m^3)
    g = 9.81        # Gravitational acceleration (m/s^2)

    options.swe_timestepper_options.solver_parameters = {
      "snes_type": "newtonls",
      "snes_linesearch_type": "bt",
      'snes_rtol': 1e-4,
      'ksp_rtol': 1e-4,
      'ksp_type': 'gmres',
      'pc_type': 'fieldsplit',
   }
    
    options.update(model_options)
    print_output(f"Exporting to {options.output_directory}")
    solver_obj.create_function_spaces()

    # Set up gauges
    station_coords = {}
    points = []

    if store_station_time_series:
        for name, data in read_station_data().items():
            sta_lat, sta_lon = data["latlon"]
            sta_x, sta_y, sta_z = coords_lonlat_to_xyz(sta_lon, sta_lat)
            
            # Track the original intended coordinates for matching later
            station_coords[name] = numpy.array([sta_x, sta_y, sta_z])
            points.append([sta_x, sta_y, sta_z])

    # Batch-filter the points using a single VertexOnlyMesh
    vom = VertexOnlyMesh(mesh2d, points, missing_points_behaviour='warn')

    # MPI-Safe Step: Gather all surviving points across all ranks ? 
    # This ensures every rank loops over the EXACT same list of valid stations.
    local_points = vom.coordinates.dat.data_ro
    global_points_list = COMM_WORLD.allgather(local_points)

    # Flatten the gathered lists, filtering out empty arrays from ranks with no points
    valid_arrays = [arr for arr in global_points_list if arr.size > 0]
    if valid_arrays:
        global_valid_points = numpy.vstack(valid_arrays)
    else:
        global_valid_points = numpy.empty((0, 3))

    # Re-identify and register the callbacks collectively
    for point in global_valid_points:
        best_name = None
        min_dist = float('inf')
        
        # Find the matching station name by calculating the minimum distance
        for name, orig_xyz in station_coords.items():
            dist = numpy.linalg.norm(point - orig_xyz)
            if dist < min_dist:
                min_dist = dist
                best_name = name
                
        # If the point matches an original gauge position within tolerance, add it
        if min_dist < 1e-3:
            cb = TimeSeriesCallback2D(
                solver_obj,
                ["elev_2d"],
                point[0],         # x
                point[1],         # y
                best_name,        # location_name (Position 5)
                z=point[2],       # z coordinate explicitly passed as a keyword argument
                append_to_log=False,
            )
            solver_obj.add_callback(cb)
            print(f"Successfully registered diagnostic gauge: {best_name}")
    # Setup boundary conditions for coastlines
    #solver_obj.bnd_functions["shallow_water"] = {
    #    223: {"elev": Constant(0.0)}
    #}

    #tide_height_file = VTKFile("tides.pvd")

    def update_forcings(t):
        """
        Executed by Thetis at every time-step loop.
        Updates the tidal potential and SAL based on current model state.
        """
        # Extract the live water elevation field from the solver
        # (Firedrake handles parallel synchronization of this field automatically)
        current_eta = solver_obj.fields.elev_2d
        
        # Compute Equilibrium Tide + solve the PDE for SAL
        tidal_forcing.update_forcing(total_forcing_meters, current_eta, t)
        # Account for spinup
        elev_ramp = 1.0 
        if spinup:
            if t < 432000: # 5 days
                elev_ramp = t / 432000   
        
        # Convert meters of head into Pascals of pressure and assign to the solver option
        p_atm_tidal.assign(-rho_0 * g * total_forcing_meters * elev_ramp)

        #tide_height_file.write(p_atm_tidal)


    return solver_obj, start_date, update_forcings
