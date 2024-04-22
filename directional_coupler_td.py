import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import gdstk
import tidy3d as td
from tidy3d.plugins.mode import ModeSolver

n_si = 3.48  # silicon refractive index
si = td.Medium(permittivity=n_si**2)

n_sio2 = 1.44  # silicon oxide refractive index
sio2 = td.Medium(permittivity=n_sio2**2)

h = 0.22

N_mode = 4

td.config.logging_level = "WARNING"

Bx = 10
By = 1



lda0 = 1.55  # central wavelength
ldas = np.linspace(1.5, 1.6, 101)  # wavelength range of interest

freq0 = td.C_0 / lda0  # corresponding central frequency
freqs = td.C_0 / ldas  # corresponding frequency range

fwidth = 0.5 * (np.max(freqs) - np.min(freqs))  # width of the excitation spectrum



def make_sim(pol, w_access, w_bus, gap, l_couple):

    # construct the access waveguide including the bends
    y = By + (w_access + w_bus) / 2 + gap
    access_wg = gdstk.RobustPath(
        (-3 * l_couple, y), w_access, simple_path=True, layer=1, datatype=0
    )
    access_wg.segment((-l_couple / 2 - Bx, y))
    access_wg.segment(
        (-l_couple / 2, y), offset=lambda u: (np.cos(u * np.pi) - 1) * By / 2
    )
    access_wg.segment((l_couple / 2, y))
    access_wg.segment(
        (l_couple / 2 + Bx, y), offset=lambda u: (np.cos((1 - u) * np.pi) - 1) * By / 2
    )
    access_wg.segment((3 * l_couple, y))

    # construct the bus waveguide
    bus_wg = gdstk.FlexPath(
        [(-3 * l_couple, 0), (3 * l_couple, 0)], w_bus, layer=1, datatype=1
    )

    # define a cell
    cell = gdstk.Cell("directional_coupler")
    cell.add(access_wg)
    cell.add(bus_wg)

    # construct a list of polyslab from the cell
    DC = td.PolySlab.from_gds(
        cell,
        gds_layer=1,
        axis=2,
        slab_bounds=(-h / 2, h / 2),
    )
    # define access waveguide and bus waveguide structures
    access_wg = td.Structure(geometry=DC[0], medium=si)
    bus_wg = td.Structure(geometry=DC[1], medium=si)

    # y coordinate of the access waveguide input
    y_in = (w_access + w_bus) / 2 + gap + By

    # simulation domain size
    Lx = l_couple + 2 * Bx + lda0
    Ly = 2 * (y_in + lda0)
    Lz = 10 * h
    sim_size = (Lx, Ly, Lz)

    # symmetry for each polarization
    if pol == "TE":
        symmetry = symmetry = (0, 0, 1)
    elif pol == "TM":
        symmetry = symmetry = (0, 0, -1)
    else:
        symmetry = symmetry = (0, 0, 0)

    # define a mode source to lauch either te0 or tm0 mode to the access waveguide
    mode_source = td.ModeSource(
        center=(-Lx / 2 + lda0 / 2, y_in, 0),
        size=(0, 6 * w_access, 8 * h),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        direction="+",
        mode_spec=td.ModeSpec(num_modes=1, target_neff=n_si),
        mode_index=0,
    )

    # define a flux monitor to measure the transmission to the bus waveguide
    bus_flux_monitor = td.FluxMonitor(
        center=(Lx / 2 - lda0 / 2, 0, 0),
        size=(0, 2 * w_bus, 6 * h),
        freqs=freqs,
        name="bus_flux",
    )

    # define a field monitor to visualize the field distribution in the xy plane
    field_monitor = td.FieldMonitor(
        center=(0, 0, 0), size=(td.inf, td.inf, 0), freqs=[freq0], name="field"
    )

    # define a mode monitor to check the mode composition at the bus waveguide
    bus_mode_monitor = td.ModeMonitor(
        center=(Lx / 2 - lda0 / 2, 0, 0),
        size=(0, 2 * w_bus, 6 * h),
        freqs=freqs,
        mode_spec=td.ModeSpec(num_modes=4, target_neff=n_si),
        name="bus_mode",
    )

    run_time = 2e-12  # simulation run time

    # define simulation
    sim = td.Simulation(
        size=sim_size,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=15, wavelength=lda0),
        structures=[access_wg, bus_wg],
        sources=[mode_source],
        monitors=[bus_flux_monitor, field_monitor, bus_mode_monitor],
        run_time=run_time,
        boundary_spec=td.BoundarySpec.all_sides(
            boundary=td.PML()
        ),  # pml is applied to all boundaries
        medium=sio2,  # the background medium is set to sio2 to model the substrate and top cladding
        symmetry=symmetry,
    )

    return sim




design_params = {
    "TM1": {"w_access": 0.4, "w_bus": 1.035, "gap": 0.3, "l_couple": 4.6},
    "TM2": {"w_access": 0.4, "w_bus": 1.695, "gap": 0.3, "l_couple": 6.8},
    "TM3": {"w_access": 0.4, "w_bus": 2.363, "gap": 0.3, "l_couple": 9},
    "TE1": {"w_access": 0.4, "w_bus": 0.835, "gap": 0.2, "l_couple": 15.5},
    "TE2": {"w_access": 0.406, "w_bus": 1.29, "gap": 0.2, "l_couple": 21.3},
    "TE3": {"w_access": 0.379, "w_bus": 1.631, "gap": 0.2, "l_couple": 17.6},
}

sim = make_sim("TE", **design_params["TE3"])

# ax = sim.plot(z=0)
# ax.set_aspect("auto")



# define mode solver
mode_solver = ModeSolver(
    simulation=sim,
    plane=td.Box(
        center=sim.monitors[0].center,
        size=sim.monitors[0].size,
    ),
    mode_spec=td.ModeSpec(num_modes=4, target_neff=n_si),
    freqs=[freq0],
)

mode_data = mode_solver.solve()
mode_data.to_dataframe()
print(mode_data)


mode_indices = [0, 1, 2, 3]

f, ax = plt.subplots(4, 1, tight_layout=True, figsize=(5, 8))

for i, mode_index in enumerate(mode_indices):
    abs(mode_data.Ey.isel(mode_index=mode_index)).plot(
        x="y", y="z", ax=ax[i], cmap="magma"
    )
    ax[i].set_title(f"|Ey(x, y)| of the TE{i} mode")

plt.show()

