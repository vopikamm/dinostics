# Module containing the DINO experiment class collecting all diagnostics.
import xarray       as xr
import xgcm         as xg
import numpy        as np
import scipy.sparse as sparse
import scipy.sparse.linalg as la

class Diagnostics:
    """ Diagnostics helper class collecting all diagnostic methods for DINO Experiment class. """
    def __init__(self, experiment):
        self.experiment = experiment

    def get_T_star(self):
        """ Compute the temperature restoring profile. """
        data        = self.experiment.data['T_2D']
        namelist    = self.experiment.namelist
        mask        = self.experiment.mask
        
        if data is not None:
            T_star = data.sst - (data.qns + data.qsr) / namelist['namusr_def']['rn_trp']
            return(T_star.where(mask.tmask.isel(z_c=0) == 1.))
        else:
            print("T_2D data not available.")
            return(None)
        
    def get_S_star(self):
        """ Compute the salinity restoring profile. """
        data        = self.experiment.data['T_2D']
        namelist    = self.experiment.namelist
        mask        = self.experiment.mask
        domain      = self.experiment.domain
        
        if data is not None:
            if 'saltflx' in list(data.keys()):
                S_star = data.sss - data.saltflx  / namelist['namusr_def']['rn_srp']
                return(S_star.where(mask.tmask.isel(z_c=0) == 1.))
            else:
                print('Warning: saltflux is not in the dataset. Assumed shape by Romain Caneill (2022).')
                return(37.12 * np.exp(- domain.gphit**2 / 260.**2 ) - 1.1 * np.exp( - domain.gphit**2 / 7.5**2 ))
        else:
            print("T_2D data not available.")
            return(None)
    
    def get_rho_star(self):
        """
        Compute the density restoring profile from salinity and temperature restoring.
        Referenced to surface pressure.
        """
        namelist    = self.experiment.namelist
        T_star = self.get_T_star()
        S_star = self.get_S_star()
        nml = namelist['nameos']
        if T_star and S_star:
            if nml['ln_seos']:
                rho_star = (
                    - nml['rn_a0'] * (1. + 0.5 * nml['rn_lambda1'] * ( T_star - 10.)) * ( T_star - 10.)
                    + nml['rn_b0'] * (1. - 0.5 * nml['rn_lambda2'] * ( S_star - 35.)) * ( S_star - 35.)
                    - nml['rn_nu'] * ( T_star - 10.) * ( S_star - 35.)
                ) + 1026
                return(rho_star)
            else:
                raise Exception('Only S-EOS has been implemented yet.')
        else:
            print("T_2D data not available.")
            return(None)

    def get_fluxes(self, d=20, tolerance=1e-20, prnt=False):
        """
        Compute:
            Qtot-field (total heat flux Qtot = Qns + Qsr + EmP * SST * c_p).
                --> c_p = 3991.86795711963 (from eosbn2)
            EmP-field  (Evaporation - Precipitation).
        To ensure zero heat/volume flux over the whole domain.
        The mean is removed from the maximum values in the tropics
        with a tapering such that the prescribed heat flux in
        high latitudes does not alter deep water formation.

        Tapering:  F(y) = A * (cos( pi * y / d ))    if |y| <= d
                        = 0                          else
    
        with        A   = qtot_mean * L / (2 * d)
        """
        data        = self.experiment.data['T_2D']
        mask        = self.experiment.mask
        domain      = self.experiment.domain
        grid        = self.experiment.grid
        if data is not None:
            c_p         = 3991.86795711963
            qns         = data.qns.where(mask.tmask.isel(z_c=0)==1.0)
            qsr         = data.qsr.where(mask.tmask.isel(z_c=0)==1.0)
            saltflx     = data.saltflx.where(mask.tmask.isel(z_c=0)==1.0)
            sss         = data.sss.where(mask.tmask.isel(z_c=0)==1.0)
            sst         = data.sss.where(mask.tmask.isel(z_c=0)==1.0)
            emp         = (saltflx / sss)

            emp_mean    = grid.average(emp, ['X', 'Y'])
            taper       = ( 
                (abs(domain.gphit) <= 20)                  # one in tropics, zero elsewhere
                * (np.cos( np.pi * domain.gphit / d) + 1)  # tapering-function
            )
            taper_mean  = grid.average(taper, ['X', 'Y'])

            while abs(emp_mean).max().values > tolerance:  
                emp_star    = emp  - emp_mean  * taper / taper_mean
                emp         = emp_star
                emp_mean    = grid.average(emp_star, ['X', 'Y'])
                if prnt:
                    print('emp_mean:    ', abs(emp_mean).max().values)

            qtot        = qns + qsr + emp_star * sst * c_p

            qtot_mean   = grid.average(qtot, ['X', 'Y'])

            while abs(qtot_mean).max().values > tolerance:  
                qtot_star   = qtot - qtot_mean * taper / taper_mean
                qtot        = qtot_star
                qtot_mean   = grid.average(qtot_star, ['X', 'Y'])
                if prnt:
                    print('qtot_mean:   ', abs(qtot_mean).max().values)

            return(qtot_star, emp_star)
        else:
            print("T_2D data not available.")
            return(None)

    def get_emp(self, d=20, tolerance=1e-20):
        """
        Compute the EmP-field (Evaporation - Precipitation).
        To ensure a conserved volume the global mean EmP should be zero.
        The mean is removed from the maximum precipitation in the tropics
        with a tapering such that the prescribed freshwater flux in
        high latitudes does not alter deep water formation.

        Tapering:  F(y) = A * (cos( pi * y / d ))    if |y| <= d
                        = 0                          else
    
        with        A   = emp_mean * L / (2 * d)
        """
        data        = self.experiment.data['T_2D']
        if data is not None:

            mask        = self.experiment.mask
            domain      = self.experiment.domain
            grid        = self.experiment.grid

            saltflx     = data.saltflx.where(mask.tmask.isel(z_c=0)==1.0)
            sss         = data.sss.where(mask.tmask.isel(z_c=0)==1.0)
            emp         = (saltflx / sss)

            emp_mean    = grid.average(emp, ['X', 'Y'])
            taper       = ( 
                (abs(domain.gphit) <= 20)                  # one in tropics, zero elsewhere
                * (np.cos( np.pi * domain.gphit / d) + 1)  # tapering-function
            )
            taper_mean  = grid.average(taper, ['X', 'Y'])

            while abs(emp_mean).max().values > tolerance:  
                emp_star    = emp  - emp_mean  * taper / taper_mean
                emp         = emp_star
                emp_mean    = grid.average(emp_star, ['X', 'Y'])
            return(emp_star)
        else:
            print("T_2D data not available.")
            return(None)

    def get_buoyancy_flux(self):
        """
        Compute buoyancy flux following Romain Caneill 2022.
        """
        g       = 9.81
        c_p     = 3991.86795711963
        rho_0   = 1026.
        data    = self.experiment.data['T_2D']

        nml     = self.experiment.namelist['nameos']
        mask    = self.experiment.mask
        # masking of SST,SSS
        sst     = data.sst.where(mask.tmask.isel(z_c=0) == 1.)
        q_tot   = (data.qns + data.qsr).where(mask.tmask.isel(z_c=0) == 1.)
        saltflx = data.saltflx.where(mask.tmask.isel(z_c=0) == 1.)

        if nml['ln_seos']:
            alpha   = nml['rn_a0'] * (1. + nml['rn_lambda1'] * ( sst - 10.) ) / rho_0
            beta    = nml['rn_a0'] / rho_0
        else:
            raise Exception('Only S-EOS has been implemented yet.')

        buoyancy_flux = g * alpha * q_tot / rho_0 / c_p - g * beta * saltflx / rho_0
        return(buoyancy_flux)
        
    
    def get_rho(self, T=None, S=None, z=0.):
        """
        Compute potential density referenced to the surface according to the EOS. 
        Uses gdepth_0 as depth for simplicity and allows only for S-EOS currently.

            z = 0.          Reference pressure level [m]
        """
        if T is None:
            try:
                print("T not provided. Use data[T_3D].toce")
                T = self.experiment.data['T_3D'].toce
            except:
                print("T_3D data not available.")
            if T is None:
                print("No temperature data available.")
                return(None)

        if S is None:
            try:
                print("S not provided. Use data[T_3D].soce")
                S = self.experiment.data['T_3D'].soce
            except:
                print("T_3D data not available.")
            if S is None:
                print("No salinity data available.")
                return(None)
        
        nml    = self.experiment.namelist['nameos']
        mask        = self.experiment.mask
        # masking of T,S
        soce = S.where(mask.tmask == 1.)
        toce = T.where(mask.tmask == 1.)
        if nml['ln_seos']:
            rho = (
                - nml['rn_a0'] * (1. + 0.5 * nml['rn_lambda1'] * ( toce - 10.) + nml['rn_mu1'] * z) * ( toce - 10.) 
                + nml['rn_b0'] * (1. - 0.5 * nml['rn_lambda2'] * ( soce - 35.) - nml['rn_mu2'] * z) * ( soce - 35.) 
                - nml['rn_nu'] * ( toce - 10.) * ( soce - 35.)
            ) + 1026
            return(rho)
        else:
            raise Exception('Only S-EOS has been implemented yet.')
        return(rho)
        
        
    def get_N_squared(self):
        """
        Compute the squared Brunt-Väisälä frequency according to the EOS. 
        Only for S-EOS currently.
        """
        data_T = self.experiment.data['T_3D']
        data_W = self.experiment.data['W_3D']
        
        if data_T is not None and data_W is not None:
            nml    = self.experiment.namelist['nameos']
            mask   = self.experiment.mask
            domain = self.experiment.domain
            grid   = self.experiment.grid
            # masking of T,S
            soce = data_T.soce.where(mask.tmask == 1.)
            toce = data_T.toce.where(mask.tmask == 1.)
            z    = domain.gdept_0.where(mask.tmask == 1.)
            if nml['ln_seos']:
                alpha   = ( nml['rn_a0'] * (1. + nml['rn_lambda1'] * ( toce - 10.) + nml['rn_mu1'] * z) + nml['rn_nu'] * soce ) / 1026.  
                beta    = ( nml['rn_b0'] * (1. - nml['rn_lambda2'] * ( soce - 35.) - nml['rn_mu2'] * z) + nml['rn_nu'] * toce ) / 1026.
                Nsq   = 9.80665 * (
                    - grid.interp(alpha, 'Z', boundary='extend')           # alpha on W
                    * grid.diff(toce, 'Z', boundary='extend')              # dT/dz
                    + grid.interp(beta, 'Z', boundary='extend')            # beta on W
                    * grid.diff(soce, 'Z', boundary='extend')              # dS/dz
                ) / data_W.e3w
                Nsq = Nsq.where(Nsq >= 1e-8).fillna(1e-7)
                return(Nsq)
            else:
                raise Exception('Only S-EOS has been implemented yet.')
        else:
            print("T_3D or W_3D data not available.")
            return(None)
        
    
    def get_BSF(self):
        """ Compute the Barotropic Streamfunction. """
        data = self.experiment.data['U_3D']
        
        if data is not None:
            domain = self.experiment.domain
            grid   = self.experiment.grid
            u_on_f  = grid.interp(data.uoce, 'Y')
            e3_on_f = grid.interp(data.e3u, 'Y')
            # Vertical integral
            U = (u_on_f * e3_on_f).sum('z_c')
            # Cumulative integral over y
            bts = (U[:,::-1,:] * domain.e2f[::-1,:]).cumsum('y_f') / 1e6
            return(bts)
        else:
            print("U_3D data not available.")
            return(None)
    
    def get_MOC(self, var, isel={'t':-1}, z=2000):
        """ Compute the Meridional Overturning Streamfunction of transport variable `var`. """
        # Prepare the meridional transport:
        domain = self.experiment.domain
        if var.name == 'vocetr_eff':
            var = var
        else:
            var = (var * domain.e3v_0 * domain.e1v)
        var_tra = self.experiment.grid_manipulation.transform_to_density(var=var, isel=isel, z=z)
        moc = var_tra.sum(dim='x_c')[...,::-1].cumsum('rho') / 1e6
        moc = moc.assign_coords(dict({'y_f': domain.gphif.isel(x_f=0).values}))
        return(moc)
    
    def get_ACC(self):
        """
        Compute the Antarctic Circumpolar Current.

        Defined as the volume transport through the periodic channel.        
        """
        data = self.experiment.data['U_3D']
        
        if data is not None:
            domain = self.experiment.domain
            acc = (data.uoce.isel(x_f=0) * domain.e3u_0.isel(x_f=0) * domain.e2u.isel(x_f=0)).sum(['y_c', 'z_c']) / 1e6
            return(acc)
        else:
            print("U_3D data not available.")
            return(None)
        
        return(acc)
        
    def get_meridional_heat_transport(self, T=None, V=None, rho=None):
        """
        Compute the meridional heat transport.

        Defined as c_p * \int \int rho * T * V dx dz.
        """
        domain = self.experiment.domain
        metrics = {
            ('X',): ['e1t', 'e1u', 'e1v', 'e1f'], # X distances
            ('Y',): ['e2t', 'e2u', 'e2v', 'e2f'], # Y distances
            ('Z',): ['e3t_0', 'e3u_0', 'e3v_0', 'e3f_0', 'e3w_0'], # Z distances
        }
        grid = xg.Grid(domain, metrics=metrics, periodic=False)

        c_p = 3991.86795711963 # From NEMO
        T_on_V = grid.interp(T, 'Y')
        rho_on_V = grid.interp(rho, 'Y')
        mht    = c_p * grid.integrate(rho_on_V * T_on_V * V, ['X', 'Z'])
        return(mht)

    def get_ocean_heat_content(self, T=None):
        """
        Compute the global ocean heat content (OHT).

        Defined as rho * c_p * \int T dV.
        """
        domain = self.experiment.domain
        mask   = self.experiment.mask.tmask
        metrics = {
            ('X',): ['e1t', 'e1u', 'e1v', 'e1f'], # X distances
            ('Y',): ['e2t', 'e2u', 'e2v', 'e2f'], # Y distances
            ('Z',): ['e3t_0', 'e3u_0', 'e3v_0', 'e3f_0', 'e3w_0'], # Z distances
        }
        grid = xg.Grid(domain, metrics=metrics, periodic=False)

        c_p   = 3991.86795711963 # From NEMO
        rho_0 = 1026.
        OHT    = rho_0 * c_p * grid.integrate(mask * T, ['X', 'Y', 'Z'])
        return(OHT)
    
    @staticmethod
    def _get_dynmodes(Nsq, e3t, e3w, nmodes=2):
        """
        Calculate the 1st nmodes ocean dynamic vertical modes.
        Based on
        http://woodshole.er.usgs.gov/operations/sea-mat/klinck-html/dynmodes.html
        by John Klinck, 1999.
        """
        # 2nd derivative matrix plus boundary conditions
        Ndz     = (Nsq * e3w)
        e3t     = e3t
        #Ndz_m1  = np.roll(Ndz, -1)
        #e3t_p1  = np.roll(e3t, 1)
        d0  = np.r_[1. / Ndz[1] / e3t[0],
                   (1. / Ndz[2:-1] + 1. / Ndz[1:-2]) / e3t[1:-2],
                   1. / Ndz[-2] / e3t[-2]]
        d1  = np.r_[0., -1. / Ndz[1:-1] / e3t[1:-1]]
        dm1 = np.r_[-1. / Ndz[1:-1] / e3t[0:-2], 0.]
        diags = 1e-4 * np.vstack((d0, d1, dm1))
        d2dz2 = sparse.dia_matrix((diags, (0, 1, -1)), shape=(len(Nsq)-1, len(Nsq)-1))
        # Solve generalized eigenvalue problem for eigenvalues and vertical
        # Horizontal velocity modes
        try:
            eigenvalues, modes = la.eigs(d2dz2, k=nmodes+1, which='SM')
            mask = (eigenvalues.imag == 0) & (eigenvalues >= 1e-10)
            eigenvalues = eigenvalues[mask]
            # Sort eigenvalues and modes and truncate to number of modes requests
            index = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[index[:nmodes]].real
            # Modal speeds
            ce = 1 / np.sqrt(eigenvalues * 1e4)
        except:
            ce = -np.ones(nmodes)

        return(ce)

    def get_vmodes(self, nmodes=2): #TODO: Not adapted for z-coords yet
        """ compute vertical modes
        Wrapper for calling `compute_vmodes` with DataArrays through apply_ufunc. 
        z levels must be in descending order (first element is at surface, last element is at bottom) with algebraic depth (i.e. negative)
        Normalization is performed here (int_z \phi^2 \dz = Hbot)

        Parameters:
        ___________
        nmodes: int, optional
            number of vertical baroclinic modes (barotropic is added)

        Returns:
        ________
        xarray.DataSet: vertical modes (p and w) and eigenvalues
        !! (currently only eigenvalues)
        _________
        """
        data_T  = self.experiment.data['T_3D']
        data_W  = self.experiment.data['W_3D']
        Nsq     = (self.get_N_squared())
        
        if Nsq is not None:
            mask = self.experiment.mask
            res = xr.apply_ufunc(self._get_dynmodes, 
                             Nsq.isel(x_c=slice(1,-1), y_c=slice(1,-1), t_y=-1).chunk({'z_f':-1}),
                             data_T.e3t.where(self.mask.tmask==1.0).isel(x_c=slice(1,-1), y_c=slice(1,-1), t_y=-1).chunk({'z_c':-1}),
                             data_W.e3w.isel(x_c=slice(1,-1), y_c=slice(1,-1), t_y=-1).chunk({'z_f':-1}),
                             input_core_dims=[['z_f'],['z_c'],['z_f']],
                             dask='parallelized', vectorize=True,
                             output_dtypes=[Nsq.dtype],
                             output_core_dims=[["mode"]],
                             dask_gufunc_kwargs={"output_sizes":{"mode":nmodes}}
                            )
            return res
        else:
            print("W_3D or T_3D data not available.")
            return(None)
    
    def get_Zanna_Bolton(self, gamma=1.0):
        """
        Implementation of the Zanna & Bolton (2020) subgrid closure discovered by a machine learning algorithm.
        The discretization of its operators follows Pavel Perezhogin.
        """
        data_U = self.experiment.data['U_3D']
        data_V = self.experiment.data['V_3D']
        data_T = self.experiment.data['T_3D']
        
        if data_U is not None and data_V is not None and data_T is not None:
            mask   = self.experiment.mask
            domain = self.experiment.domain
            grid   = self.experiment.grid

            dudx        = grid.diff(data_U.uoce * mask.umask / domain.e2u, 'X') * domain.e2t / domain.e1t
            dvdy        = grid.diff(data_V.voce * mask.vmask / domain.e1v, 'Y') * domain.e1t / domain.e2t

            dudy        = grid.diff(data_U.uoce / domain.e1u, 'Y') * domain.e1f / domain.e2f * mask.fmask
            dvdx        = grid.diff(data_V.voce / domain.e2v, 'X') * domain.e1f / domain.e1f * mask.fmask

            sh_xx       = dudx - dvdy       # Stretching deformation \tilde{D} on T-point
            sh_xy       = dvdx + dudy       # Shearing deformation D on F-point 
            vort_xy     = dvdx - dudy       # Relative vorticity \Zeta on F-point

            kappa_t     = domain.e2t * domain.e1t * mask.tmask * gamma
            kappa_f     = domain.e2f * domain.e1f * mask.fmask * gamma

            # Interpolating defomation and vorticity on opposite grid-points
            # TODO: different discretizations of the interpolation as proposed by Pavel
            vort_xy_t   = grid.interp(vort_xy,['X', 'Y']) * mask.tmask
            sh_xy_t     = grid.interp(sh_xy,['X', 'Y']) * mask.tmask
            sh_xx_f     = grid.interp(sh_xx,['X', 'Y']) * mask.fmask

            # Hydrostatic component of Txx/Tyy
            sum_sq      = 0.5 * (vort_xy_t**2 + sh_xy_t**2 + sh_xx**2)
            # Deviatoric component of Txx/Tyy        
            vort_sh     = vort_xy_t * sh_xy_t

            Txx         = - kappa_t * (- vort_sh + sum_sq)
            Tyy         = - kappa_t * (+ vort_sh + sum_sq)
            Txy         = - kappa_f * (vort_xy * sh_xx_f)

            ZB2020u     = (grid.diff(Txx * data_T.e3t * domain.e2t**2, 'X') / domain.e2u     \
                    + grid.diff(Txy * domain.e3f * domain.e1f**2, 'Y') / domain.e1u)         \
                    / (domain.e1u * domain.e2u) / (data_U.e3u + 1e-70)

            ZB2020v     = (grid.diff(Txy * data_T.e3f * domain.e2f**2, 'X') / domain.e2v      \
                    + self.grid.diff(Tyy * data_T.e3t * domain.e1t**2, 'Y') / domain.e1v)     \
                    / (domain.e1v * domain.e2v) / (data_V.e3v+1e-70)

            return {
                'ZB2020u': ZB2020u, 'ZB2020v': ZB2020v, 
                'Txx': Txx, 'Tyy': Tyy, 'Txy': Txy, 
                'sh_xx': sh_xx, 'sh_xy': sh_xy, 'vort_xy': vort_xy,
            }
        else:
            print("T_3D, U_3D or V_3D data not available.")
            return(None)
    
    def get_subgrid_forcing(self, u, v, other, factor=4, FGR=None):
        '''
        Compute subgrid forcing:
            SGSx = filter(advection) - advection(coarse_state).
        '''
        # Advection in high resolution model
        hr_adv_u, hr_adv_v = self.get_horizontal_advection(
            u=u,
            v=v,
            grid=self.experiment.grid,
            domain=self.experiment.domain,
            mask=self.experiment.mask
        )
        # Filter if FGR provided
        if FGR is not None:
            # filter scale w.r.t fine grid
            _filter_scale=FGR * factor
            print(_filter_scale)
            #filter velocities:
            u_filter = self.experiment.grid_manipulation.filter_simple_fixed_factor(
                #self,
                var=u,
                mask=self.experiment.mask.umask,
                filter_scale=_filter_scale
            )
            v_filter = self.experiment.grid_manipulation.filter_simple_fixed_factor(
                #self,
                var=v,
                mask=self.experiment.mask.vmask,
                filter_scale=_filter_scale
            )
            # filter advection
            adv_u_filter = self.experiment.grid_manipulation.filter_simple_fixed_factor(
                #self,
                var=hr_adv_u,
                mask=self.experiment.mask.umask,
                filter_scale=_filter_scale
            )
            adv_v_filter = self.experiment.grid_manipulation.filter_simple_fixed_factor(
                #self,
                var=hr_adv_v,
                mask=self.experiment.mask.vmask,
                filter_scale=_filter_scale
            )
        else:
            u_filter = u
            v_filter = v
            adv_u_filter = hr_adv_u
            adv_v_filter = hr_adv_v

        # Interpolate on T-point for coarse-graining
        u_filter_T = self.experiment.grid.interp(u_filter, 'X')
        v_filter_T = self.experiment.grid.interp(v_filter, 'Y')
        adv_u_filter_T = self.experiment.grid.interp(adv_u_filter, 'X')
        adv_v_filter_T = self.experiment.grid.interp(adv_v_filter, 'Y')

        # Coarse-graining
        u_coarse_grain = self.experiment.grid_manipulation.coarsen_weighted(
            var     = u_filter_T,
            factor  = factor
        )
        v_coarse_grain = self.experiment.grid_manipulation.coarsen_weighted(
            var     = v_filter_T,
            factor  = factor
        )
        adv_u_coarse_grain = self.experiment.grid_manipulation.coarsen_weighted(
            var     = adv_u_filter_T,
            factor  = factor
        )
        adv_v_coarse_grain = self.experiment.grid_manipulation.coarsen_weighted(
            var     = adv_v_filter_T,
            factor  = factor
        )
        #velocities on coarse grid
        u_coarse, v_coarse = self.velocities_on_coarse_grid(u_coarse_grain, v_coarse_grain, other)
        #advection on coarse grid
        adv_u_coarse, adv_v_coarse = self.velocities_on_coarse_grid(adv_u_coarse_grain, adv_v_coarse_grain, other)

        # Interpolate back to U/V-point for advection!
        u_coarse = other.grid.interp(u_coarse, 'X')
        v_coarse = other.grid.interp(v_coarse, 'Y')
        adv_u_coarse = other.grid.interp(adv_u_coarse, 'X')
        adv_v_coarse = other.grid.interp(adv_v_coarse, 'Y')
        # Compute advection on a coarse grid
        coarse_adv_u, coarse_adv_v = self.get_horizontal_advection(
            u=u_coarse,
            v=v_coarse,
            grid=other.grid,          
            domain=other.domain,      
            mask=other.mask           
        )
        # Compute subgrid forcing
        SGS_u = adv_u_coarse - coarse_adv_u
        SGS_v = adv_v_coarse - coarse_adv_v
        #TODO remote unneccessary dimensions and 
        return SGS_u, u_coarse, SGS_v, v_coarse

    @staticmethod
    def velocities_on_coarse_grid(u_on_t, v_on_t, other):
        """
        Set coarsegrained velocities on the original coarse grid.
        Currently this is only tested for 16th degree --> 4th of a degree!
        To fix: compare grids to infer hardcoded values.
        """
        # Drop dimension index coordinates, they do not match
        u_on_t = u_on_t.drop_vars(['x_c', 'y_c', 'z_c'])
        v_on_t = v_on_t.drop_vars(['x_c', 'y_c', 'z_c'])
        # Extract the outermost values along x_c
        u_x_first = u_on_t.isel(x_c=0).expand_dims('x_c', axis=-1)
        u_x_last  = u_on_t.isel(x_c=0).expand_dims('x_c', axis=-1)
        v_x_first = v_on_t.isel(x_c=0).expand_dims('x_c', axis=-1)
        v_x_last  = v_on_t.isel(x_c=0).expand_dims('x_c', axis=-1)
        # Update the longitude coordinate values
        u_x_first['glamt'] = u_x_first['glamt'] - 0.25 #TODO get from mask
        u_x_last['glamt']  = u_x_last['glamt'] + 0.25
        v_x_first['glamt'] = v_x_first['glamt'] - 0.25
        v_x_last['glamt']  = v_x_last['glamt'] + 0.25
        # Expand u_on_t along x_c
        u_expanded_x = xr.concat([u_x_first, u_on_t, u_x_last], dim='x_c', coords='minimal')
        # Expand v_on_t along x_c
        v_expanded_x = xr.concat([v_x_first, v_on_t, v_x_last], dim='x_c', coords='minimal')
        # Extract the outermost values along y_c
        u_y_first = u_expanded_x.isel(y_c=0).expand_dims('y_c', axis=-2)
        u_y_last  = u_expanded_x.isel(y_c=0).expand_dims('y_c', axis=-2)
        v_y_first = v_expanded_x.isel(y_c=0).expand_dims('y_c', axis=-2)
        v_y_last  = v_expanded_x.isel(y_c=0).expand_dims('y_c', axis=-2)
        # Update the latitude coordinate values
        u_y_first['gphit'] = u_y_first['gphit'] - 0.085585 #TODO get from mask
        u_y_last['gphit']  = u_y_last['gphit'] + 0.085585
        v_y_first['gphit'] = v_y_first['gphit'] - 0.085585
        v_y_last['gphit']  = v_y_last['gphit'] + 0.085585
        # Expand u_on_t along x_c
        u_expanded_xy = xr.concat(
            [u_y_first, u_expanded_x, u_y_last],
            dim='y_c',
            coords='minimal'
            ) * other.mask.tmask 
        # Expand v_on_t along x_c
        v_expanded_xy = xr.concat(
            [v_y_first, v_expanded_x, v_y_last],
            dim='y_c',
            coords='minimal'
            ) * other.mask.tmask
        return(u_expanded_xy.chunk({'x_c':-1, 'y_c':-1}), v_expanded_xy.chunk({'x_c':-1, 'y_c':-1}))
    
    @staticmethod
    def get_potential_vorticity(u, v, grid, domain, mask):
        """
        Compute potential vorticity on F-points. Free slip boundary condition.
        """
        q = ( 
            grid.diff(domain.e2v * v, 'X') - grid.diff(domain.e1u * u, 'Y') 
             ) * mask.fmask / domain.e1f / domain.e2f
        return(q)
    
    @staticmethod
    def get_kinetic_energy(u, v, grid, mask, hollingsworth=True):
        """
        Compute kinetic energy on T-points.
        """
        _KE = 0.5 * (grid.interp(u**2, 'X') + grid.interp(v**2, 'Y')) * mask.tmask

        if hollingsworth:
            # correction done in nemo to avoid internal instability (Hollingsworth et al. (1983))
            u_h = (u.shift(y_c=1, fill_value=0.) + u.shift(y_c=-1, fill_value=0.)) / 2.0
            v_h = (v.shift(x_c=1, fill_value=0.) + v.shift(x_c=-1, fill_value=0.)) / 2.0
            KE  = _KE * 2./3. + (grid.interp(u_h**2, 'X') + grid.interp(v_h**2, 'Y')) / 6 * mask.tmask 
        else:
            KE  = _KE
        return(KE)

    @classmethod
    def get_horizontal_advection(cls, u, v, grid, domain, mask):
        """
        Compute horizontal advection term in momentum equations.
        Static method which requires u, v, domain and mask specificaly.
        Follows discretization in NEMO ocean engine (2022) chapter 5.2,
        energy and enstrophy conserving.
        """
        # compute potential vorticity (north-east of T-point)
        q_ne = cls.get_potential_vorticity(u, v, grid, domain, mask)
        # define vorticity neighbours (ne, nw, sw, se --> north-east, ... , south-east)
        # to my knowledge no xgcm implementation for triads shifting
        q_nw = q_ne.roll(x_f=1)
        q_sw = q_nw.shift(y_f=1, fill_value=0.)
        q_se = q_ne.shift(y_f=1, fill_value=0.)
        # compute vorticity triads on T-points:
        Q_ne = (q_se + q_ne + q_nw).swap_dims({'x_f':'x_c', 'y_f':'y_c'}) * mask.tmask / 12.
        Q_nw = (q_ne + q_nw + q_sw).swap_dims({'x_f':'x_c', 'y_f':'y_c'}) * mask.tmask / 12. 
        Q_sw = (q_nw + q_sw + q_se).swap_dims({'x_f':'x_c', 'y_f':'y_c'}) * mask.tmask / 12.
        Q_se = (q_sw + q_se + q_ne).swap_dims({'x_f':'x_c', 'y_f':'y_c'}) * mask.tmask / 12.
        #shift triads for advection on U-points
        #now orientation (ne, nw, sw, se) is w.r.t. the U-point
        Q_U_ne = Q_nw.roll(x_c=-1).swap_dims({'x_c':'x_f'})
        Q_U_nw = Q_ne.swap_dims({'x_c':'x_f'})
        Q_U_sw = Q_se.swap_dims({'x_c':'x_f'})
        Q_U_se = Q_sw.roll(x_c=-1).swap_dims({'x_c':'x_f'})
        # shift v-velocities for advection on U-points
        # now orientation (ne, nw, sw, se) is w.r.t. the U-point
        V_U_ne = (v * domain.e1v * domain.e2v).roll(x_c=-1).swap_dims({'x_c':'x_f', 'y_f':'y_c'})
        V_U_nw = (v * domain.e1v * domain.e2v).swap_dims({'x_c':'x_f', 'y_f':'y_c'})
        V_U_sw = V_U_nw.shift(y_c=1, fill_value=0.)
        V_U_se = V_U_ne.shift(y_c=1, fill_value=0.)
        # shift triads for advection on V-points
        # now orientation (ne, nw, sw, se) is w.r.t. the V-point
        Q_V_ne = Q_se.shift(y_c=-1, fill_value=0.).swap_dims({'y_c':'y_f'})
        Q_V_nw = Q_sw.shift(y_c=-1, fill_value=0.).swap_dims({'y_c':'y_f'})
        Q_V_sw = Q_nw.swap_dims({'y_c':'y_f'})
        Q_V_se = Q_ne.swap_dims({'y_c':'y_f'})
        # shift u-velocities for advection on V-points
        # now orientation (ne, nw, sw, se) is w.r.t. the V-point
        U_V_se = (u * domain.e1u * domain.e2u).swap_dims({'x_f':'x_c', 'y_c':'y_f'})
        U_V_sw = (u * domain.e1u * domain.e2u).roll(x_f=1).swap_dims({'x_f':'x_c', 'y_c':'y_f'})
        U_V_nw = U_V_sw.shift(y_f=-1, fill_value=0.)
        U_V_ne = U_V_se.shift(y_f=-1, fill_value=0.)
        # compute kinetic energy on T-point
        KE = cls.get_kinetic_energy(u, v, grid, mask, hollingsworth=True)
        # compute advection on U-point
        adv_u = + mask.umask * (
            Q_U_ne * V_U_ne + 
            Q_U_nw * V_U_nw + 
            Q_U_sw * V_U_sw + 
            Q_U_se * V_U_se
        ) / domain.e1u - grid.diff(KE, 'X')
        # compute advection on U-point
        adv_v = - mask.vmask * (
            Q_V_ne * U_V_ne + 
            Q_V_nw * U_V_nw + 
            Q_V_sw * U_V_sw + 
            Q_V_se * U_V_se
        ) / domain.e2v - grid.diff(KE, 'Y')
        #reorder dimensions to t,z,y,x
        if 't' in adv_u.dims:
            if 'z_c' in adv_u.dims:
                dims_u = ('t', 'z_c', 'y_c', 'x_f')
                dims_v = ('t', 'z_c', 'y_f', 'x_c')
            else:
                dims_u = ('t', 'y_c', 'x_f')
                dims_v = ('t', 'y_f', 'x_c')
        else:
            if 'z_c' in adv_u.dims:
                dims_u = ('z_c', 'y_c', 'x_f')
                dims_v = ('z_c', 'y_f', 'x_c')
            else:
                dims_u = ('y_c', 'x_f')
                dims_v = ('y_f', 'x_c')
        
        adv_u = adv_u.transpose(*dims_u)
        adv_v = adv_v.transpose(*dims_v)
        return(adv_u, adv_v)
    



