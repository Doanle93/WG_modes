from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box
from shapely.ops import clip_by_rect
from skfem import Basis, ElementTriP0
from skfem.io.meshio import from_meshio
from tqdm import tqdm

from femwell.maxwell.waveguide import compute_modes
from femwell.mesh import mesh_from_OrderedDict

# %% tags=["remove-stderr"]
wavelength = 1.55
num_modes = 4
widths = np.linspace(0.4, 1, 7)

all_neffs = np.zeros((widths.shape[0], num_modes))
all_te_fracs = np.zeros((widths.shape[0], num_modes))
for i, width in enumerate(tqdm(widths)):
    core = box(0, 0, width, 0.22)
    polygons = OrderedDict(
        core=core,
        box=clip_by_rect(core.buffer(1.0, resolution=4), -np.inf, -np.inf, np.inf, 0),
        clad=clip_by_rect(core.buffer(1.0, resolution=4), -np.inf, 0, np.inf, np.inf),
    )

    resolutions = {"core": {"resolution": 0.1, "distance": 1}}

    mesh = from_meshio(mesh_from_OrderedDict(polygons, resolutions, default_resolution_max=0.6))

    basis0 = Basis(mesh, ElementTriP0())
    epsilon = basis0.zeros(dtype=complex)
    for subdomain, n in {"core": 3.48, "box": 1.444, "clad": 1.444}.items():
        epsilon[basis0.get_dofs(elements=subdomain)] = n**2

    modes = compute_modes(basis0, epsilon, wavelength=wavelength, num_modes=num_modes)
    all_neffs[i] = np.real([mode.n_eff for mode in modes])
    all_te_fracs[i, :] = [mode.te_fraction for mode in modes]

# %% tags=["hide-input"]
all_neffs = np.real(all_neffs)
plt.xlabel("Width of waveguide / µm")
plt.ylabel("Effective refractive index")
plt.ylim(1.444, np.max(all_neffs) + 0.1 * (np.max(all_neffs) - 1.444))
for lams, te_fracs in zip(all_neffs.T, all_te_fracs.T):
    plt.plot(widths, lams)
    plt.scatter(widths, lams, c=te_fracs, cmap="cool")
plt.colorbar().set_label("TE fraction")
plt.show()