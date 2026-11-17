import yt
from yt.frontends.gizmo.api import GizmoDataset
import yt.frontends.gadget.io  # noqa: F401 – registers gadget_hdf5 io handler
from yt.fields.derived_field import ValidateSpatial
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

def get_fn(sim_name, sim_mhdcr, sim_number, basedir="/home/suoqing/Storage/fire/"):
    fns = ['m10z_mass250/%s/output/snapdir_%03d/snapshot_%03d.0.hdf5' % (sim_mhdcr, sim_number, sim_number),
           'm11b/%s/output/snapshot_%03d.hdf5' % (sim_mhdcr, sim_number),
           'm11i_mass7100/%s/output/snapdir_%03d/snapshot_%03d.0.hdf5' % (sim_mhdcr, sim_number, sim_number),
           'm11c/%s/output/snapdir_%03d/snapshot_%03d.0.hdf5' % (sim_mhdcr, sim_number, sim_number),
           'm11d/%s/output/snapshot_%03d.hdf5' % (sim_mhdcr, sim_number),
           'm11e_mass7100/%s/output/snapdir_%03d/snapshot_%03d.0.hdf5' % (sim_mhdcr, sim_number, sim_number),
           'm11f/%s/output/snapdir_%03d/snapshot_%03d.0.hdf5' % (sim_mhdcr, sim_number, sim_number),
           'm11h/%s/output/snapshot_%03d.hdf5' % (sim_mhdcr, sim_number),
           'm12f_mass7000/%s/output/snapdir_%03d/snapshot_%03d.0.hdf5' % (sim_mhdcr, sim_number, sim_number),
           'm12i_mass7000_MHDCR_tkFIX/%s/output/snapdir_%03d/snapshot_%03d.0.hdf5' % (sim_mhdcr, sim_number, sim_number),
           'm12m_mass7000/%s/output/snapdir_%03d/snapshot_%03d.0.hdf5' % (sim_mhdcr, sim_number, sim_number),
           'm12b_mass7100/%s/output/snapdir_%03d/snapshot_%03d.0.hdf5' % (sim_mhdcr, sim_number, sim_number),
           'm12c_mass7100/%s/output/snapdir_%03d/snapshot_%03d.0.hdf5' % (sim_mhdcr, sim_number, sim_number),
           'm12r_mass7100/%s/output/snapdir_%03d/snapshot_%03d.0.hdf5' % (sim_mhdcr, sim_number, sim_number),
           'm12w_mass7100/%s/output/snapdir_%03d/snapshot_%03d.0.hdf5' % (sim_mhdcr, sim_number, sim_number),
           'm12z2_z2h600/%s/output/snapdir_%03d/snapshot_%03d.0.hdf5' % (sim_mhdcr, sim_number, sim_number),
          ]
    for fn in fns:
        if sim_name==fn[:len(sim_name)]: return basedir+fn
    if sim_name=="m10q":
        if sim_mhdcr=="mhdcv":
            fn = "m10q_mass250_MHDCR_M1_tkFIX/mhdcv/output/snapshot_%03d.hdf5" % sim_number
        elif sim_mhdcr=="cr_700":
            fn = "m10q_mass250_MHDCR_M1_tkFIX/cr_kappa1000_new2/output/snapshot_%03d.hdf5" % sim_number
        return basedir+fn
    raise ValueError("No file found with %s, %s, %d" % (sim_name, sim_mhdcr, sim_number))

def write_grid(fn, radius, fields, dims=[256,256,256], overwrite=False):

    def get_field_data(field_name, grid): # Conversion for CR energy
        if field_name == "CosmicRayEnergy":
            if not ('PartType0', 'CosmicRayEnergy') in ds.field_list: raise ValueError("No CosmicRayEnergy, skip")
            field_data = (grid["CosmicRayEnergy_spec"]).in_cgs()
        else:
            field_data = grid[field].in_cgs()
        return field_data

    f = h5py.File('%s_cut%d.hdf5' % (fn, dims[0]), "a")
    if "radius" in f.keys():
        if (f["radius"][0] != radius):
            print("radius does not match! Use radius = %f in dataset" % f["radius"][0])
            radius = f["radius"][0]

    ds = yt.load(fn) # GizmoDataset(fn)
    v, c = ds.find_max("Density")
    radius = ds.quan(radius, 'kpc')
    left = c - radius # yt knows length unit of kpc/h
    right = c + radius
    grid = ds.arbitrary_grid(left, right, dims)

    for field in fields:
        try:
            if "particle_" == field[:9]:
                field_name = field[9:]
                # delete particle_ prefix from field name
            else:
                field_name = field

            # Save O VI from hm12_ss
            # if field == "O_p5_mass_hm12_ss":
            #     field = "O_p5_mass"
            #     field_name = field
            #     import trident
            #     trident.add_ion_fields(ds, ions=["O VI"], ftype="gas")

            if field_name in f.keys():
                if not overwrite:
                    print("%s field found, skip" % field_name)
                else:
                    field_data = get_field_data(field_name, grid)
                    f[field_name][...] = field_data.v.astype("float32")
                    f.attrs["%s_units" % field_name] = str(field_data.units)
                    print("%s field updated in unit of %s" % (field_name, str(field_data.units)))
            else:
                field_data = get_field_data(field_name, grid)
                # Save O VI from hm12_ss
                # if field_name=="O_p5_mass": field_name = "O_p5_mass_hm12_ss"
                f.create_dataset(field_name, dims, 
                                 data=field_data.v.astype("float32"),
                                 dtype="float32")
                f.attrs["%s_units" % field_name] = str(field_data.units)
                print("%s field interpolated in unit of %s" % (field_name, str(field_data.units)))
        except Exception as err:
            print(err)

    if "radius" not in f.keys():
        f.create_dataset("radius", [1], data=radius.v, dtype="float64")
    if "current_redshift" not in f.keys():
        if hasattr(ds, "current_redshift"):
            f.create_dataset("current_redshift", [1], data=ds.current_redshift, dtype="float64")
    if "current_time" not in f.keys():
        if hasattr(ds, "current_time"):
            f.create_dataset("current_time", [1], data=ds.current_time.in_cgs(), dtype="float64")

    f.close()

def load_grid(fn, fields=None, exclude_fields=[], interval=1, Lbox=None):
    f = h5py.File(fn, "r")
    data = {}
    if fields is None:
        fields = f.keys()
    r = f["radius"][0]
    print("Radius of dataset: %.1f" % r)
    kpc = 3.08567758e21
    bbox = np.array([[-r, r], [-r, r], [-r, r]]) * kpc
    for field in fields:
        if field in exclude_fields: continue
        if field not in ["radius" ,"current_redshift", "current_time"]:
            if field not in f.keys():
                print("ERROR: field %s cannot be found!" % field)
                print("All fields available:", list(f.keys()))
                raise KeyError
            field_data = f[field][...]
            field_unit = f.attrs["%s_units" % field]
            if field in ["density", "temperature"]:
                if field_data.min == 0.: print("WARNING: %s field contains zero value" % field)
                field_data[field_data==0.] = field_data[field_data>0.].min()
            if field=="metallicity":
                field_data /= 0.02 # Convert from fire to solar
            data[("gas", field)] = (field_data[::interval,::interval,::interval], field_unit)
            dims = data[("gas" ,field)][0].shape
            print("%s field loaded with unit of %s" % (field, field_unit))
    if Lbox is not None: bbox=np.array([[0, Lbox], [0, Lbox], [0, Lbox]])
    ds = yt.load_uniform_grid(data, dims, length_unit="cm", bbox=bbox)
    path, filename = os.path.split(fn)
#    ds.basename = filename
    if "current_redshift" in f.keys(): ds.current_redshift = f["current_redshift"][0]
    if "current_time" in f.keys(): ds.current_time = f["current_time"][0]
    return ds

def ion_name_convert(ion_name):
    def arabic_to_roman(num):
        conv = [[1000, 'M'], [900, 'CM'], [500, 'D'], [400, 'CD'],
                [ 100, 'C'], [ 90, 'XC'], [ 50, 'L'], [ 40, 'XL'],
                [  10, 'X'], [  9, 'IX'], [  5, 'V'], [  4, 'IV'],
                [   1, 'I']]
        roman = ''
        i = 0
        while num > 0:
            while conv[i][0] > num: i+=1
            roman += conv[i][1]
            num -= conv[i][0]
        return roman

    ion_field = "%s %s" % (ion_name.split("_")[0], 
                           arabic_to_roman(int(ion_name.split("_")[1][1:])+1))
    return ion_field

def compute_sph_column(fn, my_Vars, Axes=["x", "y", "z"], overwrite=False, UVB="", one_sigma=False):

    def percentile_range(values):
        lower = np.percentile(values, 16)
        upper = np.percentile(values, 84)
        return lower, upper

    from scipy.stats import binned_statistic
    import trident

    # determine ion fields needed to be computed for all axis
    ion_Fields = []
    for axis in Axes:
        for my_var in my_Vars:
            column_fn = "%s_%s_%s_%s.npy" % (fn, my_var, axis, UVB)
            if not os.path.isfile(column_fn) or overwrite:
                if not ion_name_convert(my_var) in ion_Fields:
                    ion_Fields.append(ion_name_convert(my_var))

    if len(ion_Fields) == 0: return
    print("To be computed: %s" % ion_Fields)

    size = 800
    ds = yt.load(fn)
    trident.add_ion_fields(ds, ions=ion_Fields, ftype="gas")
    v, c = ds.find_max("density")
    r = ds.quan(1000, 'kpc')
    idx = np.arange(-size/2, size/2+1, 1)
    xx, yy = np.meshgrid(idx, idx, sparse=True)
    radius = np.sqrt(xx**2+yy**2).ravel()

    for axis in Axes:
        # determine ion fields needed to be computed for current axis
        my_Vars_compute = []
        for my_var in my_Vars:
            column_fn = "%s_%s_%s_%s.npy" % (fn, my_var, axis, UVB)
            if not os.path.isfile(column_fn) or overwrite:
                my_Vars_compute.append(("gas", my_var))
        if len(my_Vars_compute) == 0: continue
        proj = ds.proj(my_Vars_compute, axis, center=c)
        frb = proj.to_frb(r*2, size+1)
        for my_var_withtype in my_Vars_compute:
            my_var = my_var_withtype[1]
            print(my_var)
            column_fn = "%s_%s_%s_%s.npy" % (fn, my_var, axis, UVB)
            frb_data = frb[("gas", my_var)].v.ravel()
            mean_, edge, binnum = binned_statistic(radius, frb_data, statistic="mean", bins=150, range=(1,size/2))
            if one_sigma:
                min_, edge, binnum = binned_statistic(radius, frb_data, statistic=lambda x: percentile_range(x)[0], bins=150, range=(1,size/2))
                max_, edge, binnum = binned_statistic(radius, frb_data, statistic=lambda x: percentile_range(x)[1], bins=150, range=(1,size/2))
            else:
                min_, edge, binnum = binned_statistic(radius, frb_data, statistic="min", bins=150, range=(1,size/2))
                max_, edge, binnum = binned_statistic(radius, frb_data, statistic="max", bins=150, range=(1,size/2))
            med_, edge, binnum = binned_statistic(radius, frb_data, statistic="median", bins=150, range=(1,size/2))
            impact = 0.5 * (edge + np.roll(edge,1))[1:] * (2. * r.in_units("kpc").v / size)
            np.save(column_fn, np.vstack((impact, mean_, min_, max_, med_)))
            print("Column data file %s saved" % column_fn)
    return

def get_column(fn, my_var, Axes=["x", "y", "z"], overwrite=False, modifybounds=True, UVB=""):
    for enum, axis in enumerate(Axes):
        column_fn = "%s_%s_%s_%s.npy" % (fn, my_var, axis, UVB)
        if not os.path.isfile(column_fn):
            print("Missing file: %s" % column_fn)
            raise FileNotFoundError
        column_data = np.load(column_fn)
        if enum==0:
            x = column_data[0,:]; y = np.log10(column_data[4,:]); y1 = np.log10(column_data[2,:]); y2 = np.log10(column_data[3,:])
        else:
            y += np.log10(column_data[4,:]); y1 += np.log10(column_data[2,:]); y2 += np.log10(column_data[3,:])
    y /= len(Axes); y1 /= len(Axes); y2 /= len(Axes)
    if modifybounds:
        if y1.min() == -np.infty:
            print("Warning: lower bound is 0; filling the shadow area")
            y1[y1<0] = np.min([y1[y1>0].min(), y.min()])
    return x, y, y1, y2

def plot_column(x, y, y1, y2, axe=None, color=None, label=None, bounds=True, returnLine=False, x_log=False, linestyle="-"):
    if axe is None: axe = plt
    if x_log:
        line, = axe.semilogx(x, y, color=color, label=label, linewidth=2, linestyle=linestyle)
    else:
        line, = axe.plot(x, y, color=color, label=label, linewidth=2, linestyle=linestyle)
    if bounds:
        axe.fill_between(x, y1, y2, where=y2 >= y1, facecolor=color, edgecolor="none", interpolate=True, alpha=0.4)
    if returnLine:
        return axe, line
    return axe

@yt.derived_field(name=("PartType0", "CosmicRayEnergy_spec"), units="code_specific_energy", sampling_type="particle")
def _CosmicRayEnergy_spec(field,data):
    try:
        return data.ds.arr((data[("PartType0", "CosmicRayEnergy")] / data[("PartType0", "Masses")]).v, "code_specific_energy")
    except Exception as err:
        print(err)
        return data.ds.arr(np.full(data["density"].shape, np.nan), "code_specific_energy")

@yt.derived_field(name="density_cold", units="g/cm**3", sampling_type="cell")
def _density_cold(field,data):
    return data["density"] \
         * (data["temperature"].v >= 1.e4) \
         * (data["temperature"].v < 1.e5)

@yt.derived_field(name="density_warm", units="g/cm**3", sampling_type="cell")
def _density_warm(field,data):
    return data["density"] \
         * (data["temperature"].v >= 1.e5) \
         * (data["temperature"].v < 1.e6)

@yt.derived_field(name="density_hot", units="g/cm**3", sampling_type="cell")
def _density_hot(field,data):
    return data["density"] \
         * (data["temperature"].v >= 1.e6)

@yt.derived_field(name="specific_thermal_energy", units="cm**2/s**2", sampling_type="cell")
def _thermal_energy(field,data):
    try:
        return data["InternalEnergy"]
    except:
        return data.ds.arr(np.full(data["density"].shape, np.nan), "cm**2/s**2")

@yt.derived_field(name="thermal_energy_by_volume", units="erg/cm**3", sampling_type="cell")
def _thermal_energy_by_volume(field,data):
    try:
        return data["InternalEnergy"] * data["density"]
    except:
        return data.ds.arr(np.full(data["density"].shape, np.nan), "erg/cm**3")

@yt.derived_field(name="cr_energy", units="cm**2/s**2", sampling_type="cell")
def _cr_energy(field,data):
    try:
        return data["CosmicRayEnergy"]
    except Exception as err:
        print(err)
        return data.ds.arr(np.full(data["density"].shape, 0.), "cm**2/s**2")

@yt.derived_field(name="cr_energy_by_volume", units="erg/cm**3", sampling_type="cell")
def _cr_energy_by_volume(field,data):
    try:
        return data["CosmicRayEnergy"] * data["density"]
    except Exception as err:
        print(err)
        return data.ds.arr(np.full(data["density"].shape, np.nan), "erg/cm**3")

@yt.derived_field(name="cr_pressure", units="g/cm/s**2", sampling_type="cell")
def _cr_pressure(field,data):
    return data["cr_energy_by_volume"] / 3.

@yt.derived_field(name="nan_field", units="dimensionless", sampling_type="cell")
def _nan_field(field,data):
    return data.ds.arr(np.full(data["density"].shape, np.nan), "dimensionless")

def get_coolheat_fields():

    @yt.derived_field(name="Lambda",units="erg*cm**3/s", sampling_type="cell")
    def Lambda(field,data):
        dens = np.linspace(-8,1,121)
        temp = np.linspace(4,8,40)
        Z = np.array([-3, -2, -1, 0])
        Lambda = np.zeros((len(dens),len(temp),len(Z)))
        for i, Z0 in enumerate(Z):
            cooldata = np.loadtxt("Z%d.dat" % Z0)
            temp_check = cooldata[:,0]
            Lambda0 = cooldata[:,2]
            Lambda0.shape = (121,40)
            temp_check.shape = (121,40)
            if(temp_check[:,30].max()!=temp_check[:,30].min()):
                raise ValueError
            Lambda0 /= np.power(10,dens[:,None])**2.
            Lambda[:,:,i] = np.log10(Lambda0)

        n = np.log10(data["H_nuclei_density"].in_units("cm**-3").v)
        T = np.log10(data["temperature"].in_units("K").v)
        metal = np.clip(np.log10(data["metallicity"]), -3, 0) # cooling changes very little ouside this range

        from scipy.interpolate import RegularGridInterpolator
        f = RegularGridInterpolator((dens,temp,Z), Lambda, method="linear", bounds_error=False, fill_value=np.nan)
        Lambda_interp = f((n,T,metal))
        Lambda_interp = data.ds.arr(np.power(10,Lambda_interp), "erg*cm**3/s")
        return Lambda_interp

    @yt.derived_field(name="Gamma",units="erg*cm**3/s", sampling_type="cell")
    def Gamma(field,data):
        dens = np.linspace(-8,1,121)
        temp = np.linspace(4,8,40)
        Z = np.array([-3, -2, -1, 0])
        Gamma = np.zeros((len(dens),len(temp),len(Z)))
        for i, Z0 in enumerate(Z):
            heatdata = np.loadtxt("Z%d.dat" % Z0)
            temp_check = heatdata[:,0]
            Gamma0 = heatdata[:,1]
            Gamma0.shape = (121,40)
            temp_check.shape = (121,40)
            if(temp_check[:,30].max()!=temp_check[:,30].min()):
                raise ValueError
            Gamma0 /= np.power(10,dens[:,None])**2.
            Gamma[:,:,i] = np.log10(Gamma0)

        n = np.log10(data["H_nuclei_density"].in_units("cm**-3").v)
        T = np.log10(data["temperature"].in_units("K").v)
        metal = np.clip(np.log10(data["metallicity"]), -3, 0) # heating changes very little ouside this range

        from scipy.interpolate import RegularGridInterpolator
        f = RegularGridInterpolator((dens,temp,Z), Gamma, method="linear", bounds_error=False, fill_value=np.nan)
        Gamma_interp = f((n,T,metal))
        Gamma_interp = data.ds.arr(np.power(10,Gamma_interp), "erg*cm**3/s")
        return Gamma_interp

    @yt.derived_field(name="cooling_time", units="s", sampling_type="cell")
    def _cooling_time(field,data):
        k = data.ds.quan(1.380658e-16, "erg/K")
        return 3./2. * k * data["temperature"] / (data["H_nuclei_density"] * data["Lambda"])

    @yt.derived_field(name="cooling_rate", units="erg/cm**3/s", sampling_type="cell")
    def _cooling_rate(field,data):
        return data["H_nuclei_density"]**2 * data["Lambda"]

    @yt.derived_field(name="cooling_rate_volume", units="erg/s", sampling_type="cell")
    def _cooling_rate_volume(field,data):
        return data["cooling_rate"] * data["cell_volume"]

    @yt.derived_field(name="photon_heating_rate", units="erg/cm**3/s", sampling_type="cell")
    def _photon_heating_rate(field,data):
        return data["H_nuclei_density"]**2 * data["Gamma"]

    @yt.derived_field(name="net_cooling", units="erg*cm**3/s", sampling_type="cell")
    def _net_cooling_rate(field,data):
        return (data["Lambda"] - data["Gamma"])

    @yt.derived_field(name="net_heating", units="erg*cm**3/s", sampling_type="cell")
    def _net_heating_rate(field,data):
        return (data["Gamma"] - data["Lambda"])

    @yt.derived_field(name="collision_loss", units="erg/cm**3/s", sampling_type="cell")
    def _collision_loss(field,data):
        t0n0 = data.ds.quan(1.7e15, "s/cm**3")
        return data["H_nuclei_density"] * data["cr_energy_by_volume"] / t0n0

def add_abs_field(ds, abs_field):
    input_units = ds.all_data()[abs_field].units
    def _abs_field(field,data):
        return np.abs(data[abs_field])
    ds.add_field("abs_%s" % abs_field, function=_abs_field, units=input_units, sampling_type="cell")

def get_pressure_fields():

    @yt.derived_field(name='gradx_cr_pressure', units='g/cm**2/s**2',
                      validators = [ValidateSpatial(1, ['cr_pressure'])], sampling_type="cell")
    def _gradx_cr_pressure(field,data):
        data.ds.force_periodicity = (True,True,True)
        sl_left = slice(None, -2, None)
        sl_right = slice(2, None, None)
        div_fac = 2.0
        ds = div_fac * data['dx'].flat[0]
        f  = data["cr_pressure"][sl_right,1:-1,1:-1]/ds
        f -= data["cr_pressure"][sl_left ,1:-1,1:-1]/ds
        new_field = np.zeros(data["cr_pressure"].shape, dtype='float64')
        new_field = data.ds.arr(new_field,'g/cm**2/s**2')
        new_field[1:-1,1:-1,1:-1] = f
        return new_field

    @yt.derived_field(name='grady_cr_pressure', units='g/cm**2/s**2',
                      validators = [ValidateSpatial(1, ['cr_pressure'])], sampling_type="cell")
    def _grady_cr_pressure(field,data):
        data.ds.force_periodicity = (True,True,True)
        sl_left = slice(None, -2, None)
        sl_right = slice(2, None, None)
        div_fac = 2.0
        ds = div_fac * data['dy'].flat[0]
        f  = data["cr_pressure"][1:-1,sl_right,1:-1]/ds
        f -= data["cr_pressure"][1:-1,sl_left ,1:-1]/ds
        new_field = np.zeros(data["cr_pressure"].shape, dtype='float64')
        new_field = data.ds.arr(new_field,'g/cm**2/s**2')
        new_field[1:-1,1:-1,1:-1] = f
        return new_field

    @yt.derived_field(name='gradz_cr_pressure', units='g/cm**2/s**2',
                      validators = [ValidateSpatial(1, ['cr_pressure'])], sampling_type="cell")
    def _gradz_cr_pressure(field,data):
        data.ds.force_periodicity = (True,True,True)
        sl_left = slice(None, -2, None)
        sl_right = slice(2, None, None)
        div_fac = 2.0
        ds = div_fac * data['dz'].flat[0]
        f  = data["cr_pressure"][1:-1,1:-1,sl_right]/ds
        f -= data["cr_pressure"][1:-1,1:-1,sl_left ]/ds
        new_field = np.zeros(data["cr_pressure"].shape, dtype='float64')
        new_field = data.ds.arr(new_field,'g/cm**2/s**2')
        new_field[1:-1,1:-1,1:-1] = f
        return new_field

    @yt.derived_field(name="stream_loss", units="erg/cm**3/s", sampling_type="cell")
    def _stream_loss(field,data):
        vA_x = data["magnetic_field_x"] / (4.*np.pi*data["density"])**0.5
        vA_y = data["magnetic_field_y"] / (4.*np.pi*data["density"])**0.5
        vA_z = data["magnetic_field_z"] / (4.*np.pi*data["density"])**0.5
        return ((data["gradx_cr_pressure"] * vA_x)**2
               +(data["grady_cr_pressure"] * vA_y)**2
               +(data["gradz_cr_pressure"] * vA_z)**2)**0.5

    @yt.derived_field(name="radial_cr_pressure_gradient", units="g/cm**2/s**2", sampling_type="cell")
    def _radial_cr_pressure_gradient(field,data):
        len_xyz =  np.sqrt(data["x"]**2+data["y"]**2+data["z"]**2)
        unit_vec_x = data["x"] / len_xyz
        unit_vec_y = data["y"] / len_xyz
        unit_vec_z = data["z"] / len_xyz
        return (data["gradx_cr_pressure"] * unit_vec_x +
                data["grady_cr_pressure"] * unit_vec_y +
                data["gradz_cr_pressure"] * unit_vec_z)

    @yt.derived_field(name='gradx_pressure', units='g/cm**2/s**2',
                      validators = [ValidateSpatial(1, ['pressure'])], sampling_type="cell")
    def _gradx_pressure(field,data):
        data.ds.force_periodicity = (True,True,True)
        sl_left = slice(None, -2, None)
        sl_right = slice(2, None, None)
        div_fac = 2.0
        ds = div_fac * data['dx'].flat[0]
        f  = data["pressure"][sl_right,1:-1,1:-1]/ds
        f -= data["pressure"][sl_left ,1:-1,1:-1]/ds
        new_field = np.zeros(data["pressure"].shape, dtype='float64')
        new_field = data.ds.arr(new_field,'g/cm**2/s**2')
        new_field[1:-1,1:-1,1:-1] = f
        return new_field

    @yt.derived_field(name='grady_pressure', units='g/cm**2/s**2',
                      validators = [ValidateSpatial(1, ['pressure'])], sampling_type="cell")
    def _grady_pressure(field,data):
        data.ds.force_periodicity = (True,True,True)
        sl_left = slice(None, -2, None)
        sl_right = slice(2, None, None)
        div_fac = 2.0
        ds = div_fac * data['dy'].flat[0]
        f  = data["pressure"][1:-1,sl_right,1:-1]/ds
        f -= data["pressure"][1:-1,sl_left ,1:-1]/ds
        new_field = np.zeros(data["pressure"].shape, dtype='float64')
        new_field = data.ds.arr(new_field,'g/cm**2/s**2')
        new_field[1:-1,1:-1,1:-1] = f
        return new_field

    @yt.derived_field(name='gradz_pressure', units='g/cm**2/s**2',
                      validators = [ValidateSpatial(1, ['pressure'])], sampling_type="cell")
    def _gradz_pressure(field,data):
        data.ds.force_periodicity = (True,True,True)
        sl_left = slice(None, -2, None)
        sl_right = slice(2, None, None)
        div_fac = 2.0
        ds = div_fac * data['dz'].flat[0]
        f  = data["pressure"][1:-1,1:-1,sl_right]/ds
        f -= data["pressure"][1:-1,1:-1,sl_left ]/ds
        new_field = np.zeros(data["pressure"].shape, dtype='float64')
        new_field = data.ds.arr(new_field,'g/cm**2/s**2')
        new_field[1:-1,1:-1,1:-1] = f
        return new_field

    @yt.derived_field(name="radial_pressure_gradient", units="g/cm**2/s**2", sampling_type="cell")
    def _radial_pressure_gradient(field,data):
        len_xyz =  np.sqrt(data["x"]**2+data["y"]**2+data["z"]**2)
        unit_vec_x = data["x"] / len_xyz
        unit_vec_y = data["y"] / len_xyz
        unit_vec_z = data["z"] / len_xyz
        return (data["gradx_pressure"] * unit_vec_x +
                data["grady_pressure"] * unit_vec_y +
                data["gradz_pressure"] * unit_vec_z)

def get_velocity_fields():

    @yt.derived_field(name="ram_pressure", units="g/cm/s**2", sampling_type="cell")
    def _ram_pressure(field,data):
        return 0.5 * data["density"] * data["velocity_magnitude"]**2.

    # @yt.derived_field(name='gradx_ram_pressure', units='g/cm**2/s**2',
    #                   validators = [ValidateSpatial(1, ['ram_pressure'])], sampling_type="cell")
    # def _gradx_ram_pressure(field,data):
    #     data.ds.force_periodicity = (True,True,True)
    #     sl_left = slice(None, -2, None)
    #     sl_right = slice(2, None, None)
    #     div_fac = 2.0
    #     ds = div_fac * data['dx'].flat[0]
    #     f  = data["ram_pressure"][sl_right,1:-1,1:-1]/ds
    #     f -= data["ram_pressure"][sl_left ,1:-1,1:-1]/ds
    #     new_field = np.zeros(data["ram_pressure"].shape, dtype='float64')
    #     new_field = data.ds.arr(new_field,'g/cm**2/s**2')
    #     new_field[1:-1,1:-1,1:-1] = f
    #     return new_field
    #
    # @yt.derived_field(name='grady_ram_pressure', units='g/cm**2/s**2',
    #                   validators = [ValidateSpatial(1, ['ram_pressure'])], sampling_type="cell")
    # def _grady_ram_pressure(field,data):
    #     data.ds.force_periodicity = (True,True,True)
    #     sl_left = slice(None, -2, None)
    #     sl_right = slice(2, None, None)
    #     div_fac = 2.0
    #     ds = div_fac * data['dy'].flat[0]
    #     f  = data["ram_pressure"][1:-1,sl_right,1:-1]/ds
    #     f -= data["ram_pressure"][1:-1,sl_left ,1:-1]/ds
    #     new_field = np.zeros(data["ram_pressure"].shape, dtype='float64')
    #     new_field = data.ds.arr(new_field,'g/cm**2/s**2')
    #     new_field[1:-1,1:-1,1:-1] = f
    #     return new_field
    #
    # @yt.derived_field(name='gradz_ram_pressure', units='g/cm**2/s**2',
    #                   validators = [ValidateSpatial(1, ['ram_pressure'])], sampling_type="cell")
    # def _gradz_ram_pressure(field,data):
    #     data.ds.force_periodicity = (True,True,True)
    #     sl_left = slice(None, -2, None)
    #     sl_right = slice(2, None, None)
    #     div_fac = 2.0
    #     ds = div_fac * data['dz'].flat[0]
    #     f  = data["ram_pressure"][1:-1,1:-1,sl_right]/ds
    #     f -= data["ram_pressure"][1:-1,1:-1,sl_left ]/ds
    #     new_field = np.zeros(data["ram_pressure"].shape, dtype='float64')
    #     new_field = data.ds.arr(new_field,'g/cm**2/s**2')
    #     new_field[1:-1,1:-1,1:-1] = f
    #     return new_field

    # @yt.derived_field(name="radial_ram_pressure_gradient", units="g/cm**2/s**2", sampling_type="cell")
    # def _radial_ram_pressure_gradient(field,data):
    #     len_xyz =  np.sqrt(data["x"]**2+data["y"]**2+data["z"]**2)
    #     unit_vec_x = data["x"] / len_xyz
    #     unit_vec_y = data["y"] / len_xyz
    #     unit_vec_z = data["z"] / len_xyz
    #     return (data["gradx_ram_pressure"] * unit_vec_x +
    #             data["grady_ram_pressure"] * unit_vec_y +
    #             data["gradz_ram_pressure"] * unit_vec_z)

    @yt.derived_field(name="relative_velocity_x", units="cm/s", sampling_type="cell")
    def _relative_velocity_x(field,data):
        vx = np.average(data["velocity_x"], weights=data["cell_volume"])
        return (data["velocity_x"] - vx)

    @yt.derived_field(name="relative_velocity_y", units="cm/s", sampling_type="cell")
    def _relative_velocity_y(field,data):
        vy = np.average(data["velocity_y"], weights=data["cell_volume"])
        return (data["velocity_y"] - vy)

    @yt.derived_field(name="relative_velocity_z", units="cm/s", sampling_type="cell")
    def _relative_velocity_z(field,data):
        vz = np.average(data["velocity_z"], weights=data["cell_volume"])
        return (data["velocity_z"] - vz)

    @yt.derived_field(name="relative_velocity_magnitude", units="cm/s", sampling_type="cell")
    def _relative_velocity_magnitude(field,data):
        return (data["relative_velocity_x"]**2 + data["relative_velocity_y"]**2 + data["relative_velocity_z"]**2)**0.5

    @yt.derived_field(name="sph_theta", units="dimensionless", sampling_type="cell")
    def _sph_theta(field,data):
        R = np.sqrt(data["x"]**2 + data["y"]**2 + data["z"]**2)
        return data.ds.arr(np.arccos(data["z"] / R), "dimensionless")

    @yt.derived_field(name="sph_phi", units="dimensionless", sampling_type="cell")
    def _sph_phi(field,data):
        return data.ds.arr(np.arctan2(data["y"], data["x"]), "dimensionless")

    @yt.derived_field(name="relative_vr", units="cm/s", sampling_type="cell")
    def _relative_vr(field,data):
        return (data["relative_velocity_x"] * np.sin(data["sph_theta"]) * np.cos(data["sph_phi"])
              + data["relative_velocity_y"] * np.sin(data["sph_theta"]) * np.sin(data["sph_phi"])
              + data["relative_velocity_z"] * np.cos(data["sph_theta"]))

    @yt.derived_field(name="relative_vtheta", units="cm/s", sampling_type="cell")
    def _relative_vtheta(field,data):
        return (data["relative_velocity_x"] * np.cos(data["sph_theta"]) * np.cos(data["sph_phi"])
              + data["relative_velocity_y"] * np.cos(data["sph_theta"]) * np.sin(data["sph_phi"])
              - data["relative_velocity_z"] * np.sin(data["sph_theta"]))

    @yt.derived_field(name="relative_vphi", units="cm/s", sampling_type="cell")
    def _relative_vphi(field,data):
        return (-data["relative_velocity_x"] * np.sin(data["sph_phi"])
               + data["relative_velocity_y"] * np.cos(data["sph_phi"]))

    # @yt.derived_field(name="relative_radial_velocity", units="cm/s", sampling_type="cell")
    # def _relative_radial_velocity(field,data):
    #     len_xyz =  np.sqrt(data["x"]**2+data["y"]**2+data["z"]**2)
    #     unit_vec_x = data["x"] / len_xyz
    #     unit_vec_y = data["y"] / len_xyz
    #     unit_vec_z = data["z"] / len_xyz
    #     return (data["relative_velocity_x"] * unit_vec_x + 
    #             data["relative_velocity_y"] * unit_vec_y + 
    #             data["relative_velocity_z"] * unit_vec_z)

    @yt.derived_field(name="abs_relative_vr", units="cm/s", sampling_type="cell")
    def _abs_relative_vr(field,data):
        return np.abs(data["relative_vr"])

    # @yt.derived_field(name="abs_relative_tangential_velocity", units="cm/s", sampling_type="cell")
    # def _abs_relative_tangential_velocity(field,data):
    #     return np.sqrt(data["relative_velocity_magnitude"]**2 - data["relative_radial_velocity"]**2)

    @yt.derived_field(name="radial_flux", units="g/s/cm**2", sampling_type="cell")
    def _radial_flux(field,data):
        return data["relative_vr"] * data["density"]

    # @yt.derived_field(name="tangential_flux", units="g/s/cm**2", sampling_type="cell")
    # def _tangential_flux(field,data):
    #     return data["abs_relative_tangential_velocity"] * data["density"]

    @yt.derived_field(name="abs_radial_flux", units="g/s/cm**2", sampling_type="cell")
    def _abs_radial_flux(field,data):
        return np.abs(data["radial_flux"])

    @yt.derived_field(name="radial_flux_cold", units="g/s/cm**2", sampling_type="cell")
    def _radial_flux_cold(field,data):
        return data["relative_vr"] * data["density_cold"]

    @yt.derived_field(name="radial_flux_warm", units="g/s/cm**2", sampling_type="cell")
    def _radial_flux_warm(field,data):
        return data["relative_vr"] * data["density_warm"]

    @yt.derived_field(name="radial_flux_hot", units="g/s/cm**2", sampling_type="cell")
    def _radial_flux_hot(field,data):
        return data["relative_vr"] * data["density_hot"]

    @yt.derived_field(name="density_outflow", units="g/cm**3", sampling_type="cell")
    def _density_outflow(field,data):
        return data["density"] * (data["relative_vr"].v > 0.)

    @yt.derived_field(name="density_inflow", units="g/cm**3", sampling_type="cell")
    def _density_inflow(field,data):
        return data["density"] * (data["relative_vr"].v < 0.)

    @yt.derived_field(name="cell_volume_outflow", units="cm**3", sampling_type="cell")
    def _cell_volume_outflow(field,data):
        return data["cell_volume"] * (data["relative_vr"].v > 0.)

    @yt.derived_field(name="cell_volume_inflow", units="cm**3", sampling_type="cell")
    def _cell_volume_inflow(field,data):
        return data["cell_volume"] * (data["relative_vr"].v < 0.)

    @yt.derived_field(name="cell_mass_inflow", units="g", sampling_type="cell")
    def _cell_mass_inflow(field,data):
        return data["cell_mass"] * (data["relative_vr"].v < 0.)

    @yt.derived_field(name="cell_mass_outflow", units="g", sampling_type="cell")
    def _cell_mass_outflow(field,data):
        return data["cell_mass"] * (data["relative_vr"].v > 0.)

    @yt.derived_field(name="velocity_x_2", units="cm**2/s**2", sampling_type="cell")
    def _velocity_x_2(field,data):
        return data["velocity_x"]**2.

    @yt.derived_field(name="velocity_y_2", units="cm**2/s**2", sampling_type="cell")
    def _velocity_y_2(field,data):
        return data["velocity_y"]**2.

    @yt.derived_field(name="velocity_z_2", units="cm**2/s**2", sampling_type="cell")
    def _velocity_z_2(field,data):
        return data["velocity_z"]**2.

    @yt.derived_field(name="relative_vr_2", units="cm**2/s**2", sampling_type="cell")
    def _relative_vr_2(field,data):
        return data["relative_vr"]**2.

    @yt.derived_field(name="relative_vtheta_2", units="cm**2/s**2", sampling_type="cell")
    def _relative_vtheta_2(field,data):
        return data["relative_vtheta"]**2.

    @yt.derived_field(name="relative_vphi_2", units="cm**2/s**2", sampling_type="cell")
    def _relative_vphi_2(field,data):
        return data["relative_vphi"]**2.
