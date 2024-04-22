import numpy as np
import matplotlib.pyplot as plt
import gdstk

import tidy3d as td
import tidy3d.web as web

# Define simulation wavelength range

lda0 = 1.55
ldas = np.linspace(1.5, 1.6, 101)
freq0 = td.C_0 / lda0
freqs = td.C_0 / ldas
fwidth = 0.5 * (np.max(freqs) - np.min(freqs)) # frequency width of the souce


# define material
si = td.material_library["cSi"]["Li1993_293K"]
sio2 = td.material_library["SiO2"]["Horiba"]

# define geometric parameters
w = 0.5
h_si = 0.22
gap = 0.05
r = 5
inf_eff = 1e2

# simulation domain size
Lx = 2*r + 2*lda0
Ly = r / 2 + gap + 2*w + lda0
Lz = 9 * h_si

def straight_waveguide(
    x0, 
    y0, 
    z0, 
    x1, 
    y1, 
    wg_width, 
    wg_thickness, 
    medium, 
    sidewall_angle = 0):
    cell = gdstk.Cell("waveguide")

    path = gdstk.RobustPath((x0, y0), wg_width, layer = 1, datatype = 0)
    path.segment((x1, y1))

    cell.add(path)

    wg_geo = td.PolySlab.from_gds(
        cell,
        gds_layer=1,
        axis = 2,
        slab_bounds=(z0 - wg_thickness / 2, z0 + wg_thickness / 2),
        sidewall_angle=sidewall_angle,
    )

    wg = td.Structure(geometry = wg_geo[0], medium = medium)

    return wg

def ring_resonator(
    x0, 
    y0, 
    z0,
    R,
    wg_width,
    wg_thickness,
    medium,
    sidewall_angle = 0,
    ):

    cell = gdstk.Cell("top")

    path_top = gdstk.RobustPath(
        (x0 + R, y0), wg_width - wg_thickness * np.tan(np.abs(sidewall_angle)), layer =1, datatype=0
    )

    path_top.arc(R, 0, np.pi)
    cell.add(path_top)

    if sidewall_angle >= 0:
        reference_plane  = "top"
    else:
        reference_plane = "bottom"

    ring_top_geo = td.PolySlab.from_gds(
        cell,
        gds_layer = 1,
        axis = 2,
        slab_bounds = (z0 - wg_thickness / 2, z0 + wg_thickness /2),
        sidewall_angle = sidewall_angle,
        reference_plane = reference_plane,
    )

    cell = gdstk.Cell("bottom")
    path_bottom = gdstk.RobustPath(
        (x0 + R, y0), wg_width - wg_thickness * np.tan(np.abs(sidewall_angle)), layer = 1, datatype = 0)
    
    path_bottom.arc(R, 0, -np.pi)
    cell.add(path_bottom)

    ring_bottom_geo = td.PolySlab.from_gds(
        cell,
        gds_layer=1,
        axis =2,
        slab_bounds=(z0 - wg_thickness /2, z0 + wg_thickness /2),
        sidewall_angle=sidewall_angle,
        reference_plane=reference_plane,
    )

    ring = td.Structure(
        geometry=td.GeometryGroup(geometries=ring_bottom_geo + ring_top_geo), medium=medium)
    return ring



# define straight waveguide
waveguide = straight_waveguide(
    x0=-inf_eff,
    y0=0,
    z0=0,
    x1=inf_eff,
    y1=0,
    wg_width=w,
    wg_thickness=h_si,
    medium=si,
    sidewall_angle=0,
)

# define ring
ring = ring_resonator(
    x0=0,
    y0=w + gap + r,
    z0=0,
    R=r,
    wg_width=w,
    wg_thickness=h_si,
    medium=si,
    sidewall_angle=0,
)

n_si = 3.47
# mode spec for the source
mode_spec_source = td.ModeSpec(num_modes=1, target_neff=n_si)
# mode spec for the through port
mode_spec_through = mode_spec_source
# angle of the mode at the ring
theta = np.pi / 4
# mode spec for the drop port at the ring
mode_spec_drop = td.ModeSpec(
    num_modes=1, target_neff=n_si, angle_theta=theta, bend_radius=-r, bend_axis=1
)

# add a mode source as excitation
mode_source = td.ModeSource(
    center=(-r - lda0 / 4, 0, 0),
    size=(0, 6 * w, 6 * h_si),
    source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
    direction="+",
    mode_spec=mode_spec_source,
    mode_index=0,
)

# add a mode monitor to measure transmission at the through port
mode_monitor_through = td.ModeMonitor(
    center=(r + lda0 / 4, 0, 0),
    size=mode_source.size,
    freqs=freqs,
    mode_spec=mode_spec_through,
    name="through",
)

# add a mode monitor to measure transmission at the drop port
mode_monitor_drop = td.ModeMonitor(
    center=(np.sin(theta) * r, w + gap + r - np.cos(theta) * r, 0),
    size=(6 * w, 0, 6 * h_si),
    freqs=freqs,
    mode_spec=mode_spec_drop,
    name="drop",
)

# add a field monitor to visualize the field distribution
field_monitor = td.FieldMonitor(
    center=(0, 0, 0), size=(td.inf, td.inf, 0), freqs=[freq0], name="field"
)

run_time = 2e-12  # simulation run time

# construct simulation
sim_pml = td.Simulation(
    center=(0, Ly / 4, 0),
    size=(Lx, Ly, Lz),
    grid_spec=td.GridSpec.auto(min_steps_per_wvl=25, wavelength=lda0),
    structures=[waveguide, ring],
    sources=[mode_source],
    monitors=[mode_monitor_through, mode_monitor_drop, field_monitor],
    run_time=run_time,
    boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
    medium=sio2,
    symmetry=(0, 0, 1),
)

# plot simulation
# sim_pml.plot(z=0)
# plt.show()



sim_data = web.run(
    simulation=sim_pml, task_name="waveguide_to_ring", path="data/simulation_data.hdf5"
)


# ax = sim_pml.plot(z=0)
# ax.set_xlim(4, 5.5)
# ax.set_ylim(3.5, 4.5)
# plt.show()
    
    
    

