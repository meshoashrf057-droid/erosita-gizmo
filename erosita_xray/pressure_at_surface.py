import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({'font.size': 8, "mathtext.fontset": "cm"})
import yt
from firelib import *
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

@yt.derived_field(name="tot_pressure", units="eV/cm**3")
def _tot_pressure(field,data):
    return data["pressure"] + data["cr_pressure"]

def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${}e{{{}}}$'.format(a, b)

simname = "m12i"
mhdcr = "cr_700"
number = 600
my_vars = ["pressure", "cr_pressure", "tot_pressure"]
cb_title = [r"$P_\mathrm{thermal}\/\rm{(eV\/cm^{-3})}$", r"$P_\mathrm{cr}\/\rm{(eV\/cm^{-3})}$", r"$P_\mathrm{thermal}+P_\mathrm{cr}\/\rm{(eV\/cm^{-3})}$"]
R = 150
nside = 256

# fig, axes = plt.subplots(figsize=(4,9), nrows=len(my_vars))
fig = plt.figure(constrained_layout=True, figsize=(8.,5.4))
gs = GridSpec(2, 4, figure=fig)
axes = [fig.add_subplot(gs[0,0:2]), fig.add_subplot(gs[0,2:4]), fig.add_subplot(gs[1,1:3])]

plt.subplots_adjust(wspace=0.02, hspace=0.1)

fn = get_fn(simname, mhdcr, number, basedir="../data/")
ds = load_grid("%s_cut500.hdf5" % fn)

npix = hp.nside2npix(nside)
x, y, z = hp.pix2vec(nside, np.arange(npix))

x *= ds.quan(R, "kpc").in_units("code_length").v
y *= ds.quan(R, "kpc").in_units("code_length").v
z *= ds.quan(R, "kpc").in_units("code_length").v

for n, my_var in enumerate(my_vars):
    hpxmap = ds.find_field_values_at_points(my_var, ds.arr(np.vstack((x,y,z)).T, "code_length")).in_units("eV/cm**3").v

    plt.sca(axes[n])
    hp.mollview(hpxmap, cmap="arbre", hold=True, cbar=False, title=None, max=hpxmap.max()*0.9)
    fig = plt.gcf()
    ax = plt.gca()
    im = ax.get_images()[0]
    # vmin = im.get_array().min()
    # vmax = im.get_array().max()

    ax1_divider = make_axes_locatable(axes[n])
    cax = ax1_divider.append_axes("top", size="5%",pad=0)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', cmap="arbre", format=ticker.FuncFormatter(fmt))
    tick_locator = ticker.MaxNLocator(nbins=8)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.ax.set_title(cb_title[n])
    cb.ax.xaxis.set_ticks_position('top')

fig.tight_layout()
fig.savefig("pressure_surface.pdf", bbox_inches="tight")
