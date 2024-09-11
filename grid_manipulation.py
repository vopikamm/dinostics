# Module containing the DINO experiment class collecting all diagnostics.
import xarray       as xr
import xgcm         as xg
import xnemogcm     as xn
import numpy        as np
from xesmf import Regridder
from math import ceil, floor
import gcm_filters

from pathlib import Path, PurePath

class GridManipulation:
    """ GridManipulation helper class collecting grid transformations, interpolations, filtering, coarse-graining etc. """
    def __init__(self, experiment):
        self.experiment = experiment
     
    def regrid_restart(self, other):
        """Regridding a restart file to the horizontal resolution of another. """
        _lr = self.experiment.open_restart()
        lr = self.extrapolate_restart_on_land(restart_ds=_lr)
        hr = other.open_restart()

        dt = other.namelist['namdom']['rn_Dt']

        # initiate Regridder
        regridder = Regridder(lr, hr, "bilinear",  extrap_method="nearest_s2d", ignore_degenerate=True)
        restart_regrid = regridder(lr)

        # apply high resolution mask
        tmask_3D = other.mask.tmask.rename({'y_c':'y', 'x_c':'x', 'z_c':'nav_lev'}).assign_coords(dict(
            lon=(["y", "x"], other.domain.glamt.values),
            lat=(["y", "x"], other.domain.gphit.values),
            nav_lev=hr.nav_lev,
        )).drop_vars(['x', 'y']).compute()

        # 2D-variables are masked with the surface mask
        tmask_2D = tmask_3D.isel(nav_lev=0).compute()

        for var_name, var_data in restart_regrid.items():
            if var_data.ndim == 3:          #2D variables, requires including time as coordinate
                restart_regrid[var_name] = var_data.where(tmask_2D==1.0, 0.)
            if var_data.ndim == 4:          #3D variables, requires including time as coordinate
                restart_regrid[var_name] = var_data.where(tmask_3D==1.0, 0.)

        # set velocities to zero
        restart_regrid['ub'].loc[:] = 0.0
        restart_regrid['un'].loc[:] = 0.0
        restart_regrid['vb'].loc[:] = 0.0
        restart_regrid['vn'].loc[:] = 0.0

        # assign missing values from hr/lr dataset and change time_step
        restart_regrid['lon'] = hr.lon
        restart_regrid['lat'] = hr.lat
        restart_regrid['kt'] = _lr.kt
        restart_regrid['ndastp'] = _lr.ndastp
        restart_regrid['adatrj'] = _lr.adatrj
        restart_regrid['ntime'] = _lr.ntime
        restart_regrid['rdt'] = dt

        # order as the high resolution datset
        restart_regrid = restart_regrid[list(hr.keys())]

        return(restart_regrid)

    def extrapolate_restart_on_land(self, restart_ds):
        """ 
        Extrapolating a restart dataset onto land points.
        This is necessary for LR --> HR regridding.
        """
        # Mask needs to match the coordinates of the restart file
        tmask_3D = self.experiment.mask.tmask.rename({'y_c':'y', 'x_c':'x', 'z_c':'nav_lev'}).assign_coords(dict(
            lon=(["y", "x"], self.experiment.domain.glamt.values),
            lat=(["y", "x"], self.experiment.domain.gphit.values),
            nav_lev=restart_ds.nav_lev,
        )).drop_vars(['x', 'y'])

        # 2D-variables are masked with the surface mask
        tmask_2D = tmask_3D.isel(nav_lev=0)

        for var_name, var_data in restart_ds.items():
            if var_data.ndim == 3:          #2D variables, requires including time as coordinate
                restart_ds[var_name] = var_data.where(tmask_2D==1.0)
            if var_data.ndim == 4:          #3D variables, requires including time as coordinate
                restart_ds[var_name] = var_data.where(tmask_3D==1.0)
        # Fill NaN values along x
        restart_fillx   = restart_ds.interpolate_na(dim=('x'), method='nearest', fill_value="extrapolate")
        # Fill NaN values along y
        restart_fillxy  = restart_fillx.interpolate_na(dim=('y'), method='nearest', fill_value="extrapolate") 
        return(restart_fillxy)
    
    def transform_to_density(self, var, isel={'t':-1}, z=2000, levels=36):
        """Transforming a variable (vertical T-point: z_c) to density coordinates."""
        # Cut out bottom layer of z_c, such that z_f is outer (land anyway)
        ds_top  = self.experiment.domain.isel(z_c=slice(0,-1))
        var_top = var.isel(z_c=slice(0,-1), **isel)

        # Compute density if necessary
        rho = self.experiment.diagnostics.get_rho(z=z).isel(z_c=slice(0,-1), **isel).rename('rho')
        # Mask boundary
        rho = rho.where(self.experiment.mask.tmask == 1.0)
        # define XGCM grid object with outer dimension z_f 
        grid = xg.Grid(ds_top,
            coords={
                "X": {"right": "x_f", "center":"x_c"},
                "Y": {"right": "y_f", "center":"y_c"},
                "Z": {"center": "z_c", "outer": "z_f"}
            },
            metrics=xn.get_metrics(ds_top),
            periodic=False
        )

        # Interpolate sigma2 on the cell faces
        rho_var = grid.interp_like(rho, var_top).chunk({'z_c':35})      #TODO: .chunk({'z_c':-1})? 
        rho_out = grid.interp(rho_var, 'Z',  boundary='extend')

        # Target values for density coordinate
        rho_tar = np.linspace(
            floor(rho_out.min().values),
            ceil(rho_out.max().values),
            levels
        )
        # Transform variable to density coordinates:
        var_transformed = grid.transform(
            var_top,
            'Z',
            rho_tar,
            method='conservative',
            target_data=rho_out
        )
        return(var_transformed)
    
    def filter_vector(self, u, v, FGR=2):
        '''
        Algorithm:
        * Initialize GCM-filters with a given FGR:
            - "viscocity based" vector laplacian
        * Apply filter to (u,v)
        '''
        domain = self.experiment.domain
        mask   = self.experiment.mask
        # getting grid_vars necessary for c-grid gcm_filter
        # dimension names are swapped, but C-grid is assumed by gcm_filters
        grid_vars={
            # grid info centered at T-points
            'wet_mask_t'    : mask.tmask,
            'dxT'           : domain.e1t,
            'dyT'           : domain.e2t,
            # grid info centerntered at U-points
            'dxCu'          : domain.e1u.swap_dims({"x_f": "x_c"}),
            'dyCu'          : domain.e2u.swap_dims({"x_f": "x_c"}),
            'area_u'        : (domain.e1u * domain.e2u).swap_dims({"x_f": "x_c"}),
            # grid info centerntered at V-points
            'dxCv'          : domain.e1v.swap_dims({"y_f": "y_c"}),
            'dyCv'          : domain.e2v.swap_dims({"y_f": "y_c"}),
            'area_v'        : (domain.e1v * domain.e2v).swap_dims({"y_f": "y_c"}),
            # grid info centerntered at vorticity points
            'wet_mask_q'    : mask.fmask.swap_dims({"x_f": "x_c", "y_f": "y_c"}),
            'dxBu'          : domain.e1f.swap_dims({"x_f": "x_c", "y_f": "y_c"}),
            'dyBu'          : domain.e2f.swap_dims({"x_f": "x_c", "y_f": "y_c"}),
            # info about isotrsotropy of the grid -> isotropic everywhere
            'kappa_iso'     : xr.ones_like(domain.e2t),
            'kappa_aniso'   : xr.zeros_like(domain.e2t),
        }
        # minimum grid spacing
        dxmin = domain.e1t.where(mask.tmask.isel(z_c=0)==1.0).min().values # e1t = e2t in DINO
        #initialize filter
        _filter_c_grid = gcm_filters.Filter(
                filter_scale=FGR,
                dx_min=dxmin,
                filter_shape=gcm_filters.FilterShape.GAUSSIAN,
                grid_type=gcm_filters.GridType.VECTOR_C_GRID,
                grid_vars=grid_vars
                )
        # fields need same dimension names, even when staggered
        u_tmp    = u.swap_dims({"x_f": "x_c"})
        v_tmp    = v.swap_dims({"y_f": "y_c"})
        # apply filter
        (u_filtered, v_filtered) = _filter_c_grid.apply_to_vector(u_tmp, v_tmp, dims=['y_c', 'x_c'])
        #rename dimensions to original and return
        return(u_filtered.swap_dims({"x_c":"x_f"}), v_filtered.swap_dims({"y_c":"y_f"}))

    def filter_fixed_factor(self, t, FGR=4):
        '''
        Algorithm:
        * Initialize GCM-filters with a given FGR:
            - "diffusion based" laplacian
        * Apply filter to variable on T-grid
        '''
        domain = self.experiment.domain
        mask   = self.experiment.mask
        # getting grid_vars necessary for fixed gcm_filter
        dxw     = domain.e1u.swap_dims({"x_f": "x_c"})
        dyw     = domain.e2u.swap_dims({"x_f": "x_c"})
        dxs     = domain.e1v.swap_dims({"y_f": "y_c"})
        dys     = domain.e2v.swap_dims({"y_f": "y_c"})
        # maximum/minimum grid spacing
        dx_max  = max(dxw.max(),dyw.max(),dxs.max(),dys.max()).values
        dx_min  = min(dxw.min(),dyw.min(),dxs.min(),dys.min()).values
        filter_scale = FGR * dx_max
        kappa_w = dxw * dxw / (dx_max * dx_max)
        kappa_s = dys * dys / (dx_max * dx_max)
        # dimension names are swapped, but C-grid is assumed by gcm_filters
        grid_vars={
                'wet_mask': mask.tmask, 
                'dxw': dxw,
                'dyw': dyw,
                'dxs': dxs,
                'dys': dys,
                'area': (domain.e1t * domain.e2t), 
                'kappa_w': kappa_w,
                'kappa_s': kappa_s
            }
        #initialize filter
        _filter_fixed_factor = gcm_filters.Filter(
            filter_scale=filter_scale,
            dx_min=dx_min,
            filter_shape=gcm_filters.FilterShape.GAUSSIAN,
            grid_type=gcm_filters.GridType.IRREGULAR_WITH_LAND,
            grid_vars=grid_vars
        )
        # apply filter
        t_filtered = _filter_fixed_factor.apply(t, dims=['y_c', 'x_c'])
        #rename dimensions to original and return
        return(t_filtered)
    
    def filter_simple_fixed_factor(self, var, mask, filter_scale=10):
        '''
        Algorithm:
        * Initialize GCM-filters with a given FGR:
            - simple laplacian unaware of the grid
        * Apply filter to variable on any grid
        '''
        # getting grid_vars necessary for simple fixed gcm_filter
        grid_vars={
                'wet_mask': mask, 
            }
        dims=list(mask.isel(z_c=0).dims)
        #initialize filter
        _filter_simple_fixed_factor = gcm_filters.Filter(
            filter_scale=filter_scale,
            dx_min=1,
            filter_shape=gcm_filters.FilterShape.GAUSSIAN,
            grid_type=gcm_filters.GridType.REGULAR_WITH_LAND,
            grid_vars=grid_vars
        )
        # apply filter
        var_filtered = _filter_simple_fixed_factor.apply(var, dims=dims)
        #rename dimensions to original and return
        return(var_filtered)
    
    @staticmethod
    def discard_land(x, percentile=1):
        '''
        Input is the mask array. Supposed that it was
        obtained with interpolation or coarsegraining

        percentile controls how to treat land:
        * percentile=1 means that if in an averaging
        box during coarsening there was any land point,
        we treat coarse point as land point
        * percentile=0 means that of in an averaging box
        there was at least one computational point, we 
        treat coarse point as wet point
        * percentile=0.5 means that if in an averaging
        box there were more than half wet points,
        we treat coarse point as wet point
        '''
        if percentile<0 or percentile>1:
            print('Error: choose percentile between 0 and 1')
        if percentile==1:
            return (x==1).astype('float32')
        else:
            return (x>percentile).astype('float32')

    def coarsen_weighted(self, var, factor=None):
        '''
        Algorithm: 
        * Cut land and periodic row in channel.
            --> This can discard single rows of water data
            --> No exact conservational properties
        * Coarsegrain on t-points

        Note: we weight here all operations with local grid area.
        '''
        domain = self.experiment.domain
        mask   = self.experiment.mask
        grid   = self.experiment.grid

        # Hardcoded values for coarsening from 1/16 --> 1/4 degree,
        # since horizontal dimension-size depends on mercator projection    
        #TODO Could compute from resolution analyticaly and compute needed domain-size
        # 1/4° --> 1°
        # var_inner   = var.isel(x_c=slice(1,-1), y_c=slice(1,-2))
        # area_inner  = (domain.e1t * domain.e2t).isel(x_c=slice(1,-1), y_c=slice(1,-2))
        # mask_inner  = mask.tmask.isel(x_c=slice(1,-1), y_c=slice(1,-2))
        #1/4° --> 1°
        var_inner   = var.isel(x_c=slice(1,-1), y_c=slice(0,-1))
        area_inner  = (domain.e1t * domain.e2t).isel(x_c=slice(0,-1), y_c=slice(0,-1))
        mask_inner  = mask.tmask.isel(x_c=slice(1,-1), y_c=slice(0,-1))

        # coarse-graining
        coarsen = lambda x: x.coarsen({'x_c':factor, 'y_c':factor}).sum()

        # coarse graining mask and mask coarse cells with one single land point  
        mask_coarse = self.discard_land(coarsen(mask_inner * area_inner) / coarsen(area_inner), percentile=1)
        # coarse grain var
        var_coarse = coarsen(var_inner * area_inner) / coarsen(area_inner)
        # check if variable is 3D, or SURFACE variable and apply mask accordingly.
        #TODO make variable slicing possible (need to slice mask)
        if 'z_c' in var_inner.dims:   
            return(var_coarse.where(mask_coarse==1.0))
        else:
            return(var_coarse.where(mask_coarse.isel(z_c=0)==1.0))
        
        
    # def coarsen(self, factor=10, factor,
    #             coarsening=CoarsenWeighted(), filtering=Filtering(), percentile=0):
    #     '''
    #     Coarsening of the dataset with a given factor

    #     Algorithm:
    #     * Initialize coarse grid
    #     * Coarsegrain velocities by applying operator
    #     * Return new dataset with coarse velocities

    #     Note: FGR is an absolute value w.r.t. fine grid
    #           FGR_multiplier is w.r.t. coarse grid
    #     '''

    #     # Filter if needed
    #     if FGR is not None:
    #         data = xr.Dataset()
    #         data['u'], data['v'], data['rho'] = \
    #             filtering(self.data.u, self.data.v, self.state.rho(), self,
    #                         FGR) # Here FGR is w.r.t. fine grid
    #         ds_filter = DatasetCM26(data, self.param)
    #     else:
    #         ds_filter = self
        
    #     # Coarsegrain if needed
    #     if factor > 1:
    #         param = self.init_coarse_grid(factor=factor, percentile=percentile)
    #         data = xr.Dataset()
    #         ds_coarse = DatasetCM26(data, param)

    #         data['u'], data['v'], data['rho'] = \
    #             coarsening(ds_filter.data.u, ds_filter.data.v, ds_filter.state.rho(), 
    #                        ds_filter, ds_coarse, factor=factor)
    #         # To properly initialize ds_coarse.state
    #         del ds_coarse
    #         ds_coarse = DatasetCM26(data, param)
    #     else:
    #         ds_coarse = ds_filter
        
    #     return ds_coarse


