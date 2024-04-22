import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import gdstk

import tidy3d as td
import tidy3d.web as web
from tidy3d.plugins.mode import ModeSolver



td.config.logging_level = "ERROR"


lda0 = 1.55
ldas = np.linspace(1.5, 1.6, 101)

freq0 = td.C_0 / lda0
freqs = td.C_0 / ldas

fwidth = 0.5 * (np.max(freqs) - np.min(freqs))

n_si = 3.48
si = td.Medium(permittivity = n_si**2)

n_sio2 = 1.444
sio2 = td.Medium(permittivity = n_sio2**2)

h = 0.22
ws = np.linspace(0.4, 1, 7)

N_mode = 4

mode_spec = td.ModeSpec(num_modes = N_mode, target_neff = n_si)

grid_spec = td.GridSpec.auto(min_steps_per_wvl = 30, wavelength= lda0)

bound_spec = td.BoundarySpec.all_sides(boundary=td.PML())

n_eff = np.zeros((len(ws), N_mode))

for i, w in enumerate(ws):

    waveguide = td.Structure(
        geometry = td.Box(center = (0, 0, 0), size = (w, td.inf, h)),
        medium = si
    )

    sim_size = (6 * w, 1, 8 * h)
    mode_size = (6 * w, 0, 8 * h)

    sim = td.Simulation(
        size = sim_size,
        grid_spec = grid_spec,
        structures= [waveguide],
        sources=[],
        monitors = [],
        run_time = 1e-11,
        boundary_spec = bound_spec,
        medium = sio2,
        symmetry = (0, 0, 1), 
    )

    mode_solver = ModeSolver(
        simulation = sim, 
        plane = td.Box(center=(0, 0, 0), size = mode_size),
        mode_spec=mode_spec,
        freqs=[freq0]
    )

    mode_data = mode_solver.solve()

    n_eff[i] = mode_data.n_eff.values


for i in range(N_mode):
    plt.plot(ws, n_eff[:, i])

plt.ylim(1.44, 3)
plt.legend(("TE0", "TE1", "TE2", "TE3"))
plt.xlabel("Waveguide width ($'mu m$)")
plt.ylabel("Effective index")
plt.show()