# Module containing the DINO experiment class collecting all diagnostics.
import xarray       as xr
import xgcm         as xg
import xnemogcm     as xn
import cftime as cft
import f90nml
import numpy        as np
from diagnostics import Diagnostics
from grid_manipulation import GridManipulation

from pathlib import Path, PurePath

class NemoExperiment:
    """ Experiment class organizing data, grid, namelist, restarts of a DINO experiment in NEMO. """
    def __init__(self, path, experiment_name, restarts=None, files=['U_3D', 'V_3D', 'T_3D', 'W_3D', 'U_2D', 'V_2D', 'T_2D']):
        self.name           = experiment_name
        self.path           = path + experiment_name + '/'
        self.files          = files 
        self.restarts       = self.get_restarts(restarts)
        self.domain         = self.open_domain()
        self.mask           = self.open_mask()
        self.namelist       = self.open_namelist()
        self.grid           = xg.Grid(self.domain, metrics=xn.get_metrics(self.domain), periodic=False)

        self.data           = dict()
        for file in files:
            try:
                self.data[file] = self.open_data('*_grid_'+file +'*')
            except:
                self.data[file] = None
        
        # Initialize the helper classes for diagnostics and grid manipulations (Regridding, coordinate transformation, etc.)
        self.diagnostics = Diagnostics(self)
        self.grid_manipulation = GridManipulation(self)
    
    def get_restarts(self, restarts):
        """" Detect restart folders if they exist and return their names."""
        if restarts is None:
            restarts = []
            for paths in Path(self.path).iterdir():
                if paths.is_dir() and 'restart' in PurePath(paths).name:
                    restarts.append(PurePath(paths).name)
            restarts = sorted(restarts, key=lambda x: float(x.strip('restart')))
            if not restarts:
                restarts.append('')
            return(restarts)
        else:
            return(restarts)
    
    def open_domain(self, chunks='auto'):
        """ Open the domain_cfg, not lazy."""
        try:
            domain = xn.open_domain_cfg(
                files=[self.path + self.restarts[0] + '/domain_cfg.nc']
            ).chunk(chunks)
        except:
            domain = xn.open_domain_cfg(
                files=[self.path + self.restarts[0] + '/domain_cfg_out.nc']
            ).chunk(chunks)
        for var in list(domain.keys()):
            if (domain[var].dtype == 'float64'):
                domain[var] = domain[var].astype('float32')
        return(domain)

    def open_mask(self, chunks='auto'):
        """ Open the mesh_mask, lazy."""
        mask = xn.open_domain_cfg(
            files=[self.path + self.restarts[0] + '/mesh_mask.nc']
        ).chunk(chunks)
        for var in list(mask.keys()):
            if (mask[var].dtype == 'float64'):
                mask[var] = mask[var].astype('float32')
        return(mask)

    def open_data(self, file_name):
        """ Open the data lazy. """
        Data = []
        for restart in self.restarts:
            files     = Path(self.path + restart).glob(file_name)
            Data.append( xn.open_nemo(domcfg=self.domain, files=files))
        if Data:
            return(xr.concat(Data,  "t", data_vars='minimal', coords='minimal'))
        else:
           return(None)
    
    def open_namelist(self, restart=0):
        """ Open the namelist_cfg as a f90nml dict."""
        namelist = f90nml.read(
            self.path + self.restarts[restart] + '/namelist_cfg'
        )
        return(namelist)
    
    def open_restart(self, restart_path=None):
        """ Open one or multiple restart files."""
        restart_files = []
        for paths in sorted(Path(self.path).iterdir()):
            if (str(self.namelist['namrun']['nn_itend']) + '_restart.nc') in PurePath(paths).name:
                restart_files.append(PurePath(paths))

        chunks = {}
        for dim in xr.open_dataset(restart_files[0]).dims:
            chunks[dim] = 1 if dim == 'nav_lev' else -1

        ds = xr.open_mfdataset(
            restart_files,
            preprocess=xn.domcfg.domcfg_preprocess,
            combine_attrs="drop_conflicts",
            data_vars="minimal",
            drop_variables=["x", "y"],
            chunks=chunks
        )
        for i in [
            "DOMAIN_position_first",
            "DOMAIN_position_last",
            "DOMAIN_number",
            "DOMAIN_number_total",
            "DOMAIN_size_local",
        ]:
            ds.attrs.pop(i, None)
        ds = ds.assign_coords(dict(
            lon=(["y", "x"], self.domain.glamt.values),
            lat=(["y", "x"], self.domain.gphit.values)
        )).drop_vars(['x', 'y'])
        return(ds)

class OceananigansExperiment:
    """ Experiment class organizing data, grid, namelist, restarts of a DINO experiment in Oceananigans. """
    def __init__(self, path, experiment_name):
        self.name           = experiment_name
        self.path           = path + experiment_name + '/' 
        #self.domain         = self.open_domain()
        #self.mask           = self.open_mask()
        #self.grid           = xg.Grid(self.domain, metrics=xn.get_metrics(self.domain), periodic=False)

        self.data           = dict()
        for file in files:
            try:
                self.data[file] = self.open_data('*_grid_'+file +'*')
            except:
                self.data[file] = None
        
        # Initialize the helper classes for diagnostics and grid manipulations (Regridding, coordinate transformation, etc.)
        self.diagnostics = Diagnostics(self)
        self.grid_manipulation = GridManipulation(self)