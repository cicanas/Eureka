import numpy as np
import astropy.constants as const
import inspect
import os
import batman
try:
    from pyspotrod import pyspotrod as spotrod    
    print("Using C version of spotrod.")
except ImportError:
    print("Using a slower python version of spotrod. Functionality may be limited.")
    from . import Slowspotrod as spotrod

from .Model import Model
from .KeplerOrbit import KeplerOrbit
from ..limb_darkening_fit import ld_profile
from ...lib.split_channels import split

try:
    import os
    os.sys.path.append(os.environ['LDC3_PATH'])
    import LDC3
except ImportError:
    print("Could not import LDC3. Code will break if you specify an LD law of 'kipping2015'.")


def limbdarkening(r, u, ld_law = 'quadratic'):
    """
    Get the limb-darkening array for each annulus centered at radius r

    INPUT:
        r - array of radius from 0-1
        u - array of limb-darkening coefficients for law specified, assumed be of increasing coefficient order [c1, c2, c3, c4]
        ld_law - string for the given law

    OUTPUT:
        answer - intensity at specified annuli for the given law
    """    
    answer = np.zeros(r.shape) 
    mask = (r<=1.0)
    mu = np.sqrt(1.0 - r[mask]**2)
    #See https://ui.adsabs.harvard.edu/abs/2000A&A...363.1081C/abstract
    if ld_law != 'uniform':
        answer[mask] = ld_profile(ld_law).__call__(mu,*u)
    else:
        answer[mask] = 1.0
    return answer


def planet_XYZ_position(f,a,inc,ecc,omega,projobliq=0):
    """
    Get planet XYZ position, where positive Z points to the viewer
    See Winn et al. 2010

    INPUT:
        f - true anomaly in radians
        aRs - in a/R*
        inc - in radians
        ecc - eccentricity
        omega - omega in radians
        projobliq - projected obliquity (lambda) in radians

    OUTPUT:
        X - planet X position
        Y - planet Y position
        Z - planet Z position
    """
    # Equations 1-3 in Winn 2010: https://arxiv.org/pdf/1001.2010.pdf, accounting for rotation about Z
    r = a*(1.-ecc**2.)/(1.+ecc*np.cos(f))
    X = r * ( -np.cos(projobliq)*np.cos(f+omega)+np.cos(inc)*np.sin(projobliq)*np.sin(f+omega) )
    Y = r * ( -np.cos(f+omega)*np.sin(projobliq)-np.cos(inc)*np.cos(projobliq)*np.sin(f+omega) )
    # Projected separation from the center of the stellar disk
    Z = np.sqrt(X**2 + Y**2)
    return X, Y, Z


class PlanetParams():
    """
    Define planet parameters.
    """
    def __init__(self, model, pid=0, channel=0):
        """ 
        Set attributes to PlanetParams object.

        Parameters
        ----------
        model : object
            The model.eval object that contains a dictionary of parameter names 
            and their current values.
        pid : int; optional
            Planet ID, default is 0.
        channel : int, optional
            The channel number for multi-wavelength fits or mutli-white fits.
            Defaults to 0.
        """
        # Planet ID
        self.pid = pid
        if pid == 0:
            self.pid_id = ''
        else:
            self.pid_id = str(self.pid)
        # Channel ID
        self.channel = channel
        if channel == 0:
            self.channel_id = ''
        else:
            self.channel_id = f'_{self.channel}'
        # Set transit/eclipse parameters
        self.t0 = None
        self.rprs = None
        self.rp = None
        self.inc = None
        self.ars = None
        self.a = None
        self.per = None
        self.ecc = 0.
        self.w = None
        self.fp = None   
        self.t_secondary = None             
        for item in self.__dict__.keys():
            item0 = item+self.pid_id
            try:
                if model.parameters.dict[item0][1] == 'free':
                    item0 += self.channel_id
                setattr(self, item, model.parameters.dict[item0][0])
            except KeyError:
                pass
        # Allow for rp or rprs
        if (self.rprs is None) and ('rp' in model.parameters.dict.keys()):
            item0 = 'rp' + self.pid_id
            if model.parameters.dict[item0][1] == 'free':
                item0 += self.channel_id
            setattr(self, 'rprs', model.parameters.dict[item0][0])
        if (self.rp is None) and ('rprs' in model.parameters.dict.keys()):
            item0 = 'rprs' + self.pid_id
            if model.parameters.dict[item0][1] == 'free':
                item0 += self.channel_id
            setattr(self, 'rp', model.parameters.dict[item0][0])
        # Allow for a or ars
        if (self.ars is None) and ('a' in model.parameters.dict.keys()):
            item0 = 'a' + self.pid_id
            if model.parameters.dict[item0][1] == 'free':
                item0 += self.channel_id
            setattr(self, 'ars', model.parameters.dict[item0][0])
        if (self.a is None) and ('ars' in model.parameters.dict.keys()):
            item0 = 'ars' + self.pid_id
            if model.parameters.dict[item0][1] == 'free':
                item0 += self.channel_id
            setattr(self, 'a', model.parameters.dict[item0][0])
        # Allow for sampling from a unit disk
        if ('sesinw' in model.parameters.dict.keys()) and ('secosw' in model.parameters.dict.keys()):
            item0 = 'secosw' + self.pid_id
            item1 = 'sesinw' + self.pid_id
            eccval = model.parameters.dict[item0][0]**2 + model.parameters.dict[item1][0]**2
            omegaval = np.arctan2(model.parameters.dict[item1][0],model.parameters.dict[item0][0])*180./np.pi
            self.ecc = eccval
            self.w = omegaval
        # Set stellar radius
        if 'Rs' in model.parameters.dict.keys():
            item0 = 'Rs'
            if model.parameters.dict[item0][1] == 'free':
                item0 += self.channel_id
            setattr(self, 'Rs', model.parameters.dict[item0][0])


class SpotrodTransitModel(Model):
    """Transit Model"""
    def __init__(self, **kwargs):
        """Initialize the transit model

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
            Can pass in the parameters, longparamlist, nchan, and
            paramtitles arguments here.
        """
        # Inherit from Model class
        super().__init__(**kwargs)
        # Define transit model to be used
        self.transit_model = batman.TransitModel

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

        log = kwargs.get('log')

        # Store the ld_profile
        self.ld_from_S4 = kwargs.get('ld_from_S4')
        ld_func = ld_profile(self.parameters.limb_dark.value, 
                             use_gen_ld=self.ld_from_S4)
        len_params = len(inspect.signature(ld_func).parameters)
        self.coeffs = ['u{}'.format(n) for n in range(len_params)[1:]]

        self.ld_from_file = kwargs.get('ld_from_file')

        # Replace u parameters with generated limb-darkening values
        if self.ld_from_S4 or self.ld_from_file:
            log.writelog("Using the following limb-darkening values:")
            self.ld_array = kwargs.get('ld_coeffs')
            for c in range(self.nchannel_fitted):
                chan = self.fitted_channels[c]
                if self.ld_from_S4:
                    ld_array = self.ld_array[len_params-2]
                else:
                    ld_array = self.ld_array
                for u in self.coeffs:
                    index = np.where(np.array(self.paramtitles) == u)[0]
                    if len(index) != 0:
                        item = self.longparamlist[c][index[0]]
                        param = int(item.split('_')[0][-1])
                        ld_val = ld_array[chan][param-1]
                        log.writelog(f"{item}, {ld_val}")
                        # Use the file value as the starting guess
                        self.parameters.dict[item][0] = ld_val
                        # In a normal prior, center at the file value
                        if (self.parameters.dict[item][-1] == 'N' and
                                self.recenter_ld_prior):
                            self.parameters.dict[item][-3] = ld_val
                        # Update the non-dictionary form as well
                        setattr(self.parameters, item,
                                self.parameters.dict[item])
        
        # Get relevant spot parameters
        self.spotrad = np.zeros((self.nchannel_fitted, 10))
        self.spotx = np.empty((self.nchannel_fitted, 10))*np.nan
        self.spoty = np.empty((self.nchannel_fitted, 10))*np.nan  
        self.spotu = np.empty((self.nchannel_fitted, 10))*np.nan
        self.spotv = np.empty((self.nchannel_fitted, 10))*np.nan
        self.spotlat = np.empty((self.nchannel_fitted, 10))*np.nan
        self.spotlon = np.empty((self.nchannel_fitted, 10))*np.nan
        self.keys = list(self.parameters.dict.keys())
        self.keys = [key for key in self.keys if key.startswith('spot')]

    def eval(self, channel=None, pid=None, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        pid : int; optional
            Planet ID, default is None which combines the eclipse models from
            all planets.
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        lcfinal : ndarray
            The value of the model at the times self.time.
        """
        if channel is None:
            nchan = self.nchannel_fitted
            channels = self.fitted_channels
        else:
            nchan = 1
            channels = [channel, ]
        
        if pid is None:
            pid_iter = range(self.num_planets)
        else:
            pid_iter = [pid,]

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Set all parameters
        lcfinal = np.array([])
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            light_curve = np.ma.ones(time.shape)
            for pid in pid_iter:
                # Initialize planet
                pl_params = PlanetParams(self, pid, chan)

                # Set limb darkening parameters
                uarray = []
                for u in self.coeffs:
                    index = np.where(np.array(self.paramtitles) == u)[0]
                    if len(index) != 0:
                        item = self.longparamlist[chan][index[0]]
                        uarray.append(self.parameters.dict[item][0])
                pl_params.u = uarray
                pl_params.limb_dark = self.parameters.dict['limb_dark'][0]

                # Enforce physicality to avoid crashes from batman by returning
                # something that should be a horrible fit
                if not ((0 < pl_params.per) and (0 < pl_params.inc < 90) and
                        (1 < pl_params.a) and (0 <= pl_params.ecc < 1)):
                    # Returning nans or infs breaks the fits, so this was the
                    # best I could think of
                    light_curve = 1e12*np.ma.ones(time.shape)
                    continue

                # Use batman ld_profile name
                if self.parameters.limb_dark.value == 'kipping2013':
                    # Enforce physicality to avoid crashes from batman by
                    # returning something that should be a horrible fit
                    if pl_params.u[0] <= 0:
                        # Returning nans or infs breaks the fits, so this was
                        # the best I could think of
                        light_curve = 1e12*np.ma.ones(time.shape)
                        continue
                    pl_params.limb_dark = 'quadratic'
                    u1 = 2*np.sqrt(pl_params.u[0])*pl_params.u[1]
                    u2 = np.sqrt(pl_params.u[0])*(1-2*pl_params.u[1])
                    pl_params.u = np.array([u1, u2])
                elif self.parameters.limb_dark.value == 'kipping2015':
                    pl_params.limb_dark = 'nonlinear'
                    u1, u2, u3 = LDC3.forward(pl_params.u)
                    # Enforce physicality to avoid crashes from batman by
                    # returning something that should be a horrible fit
                    passed = LDC3.criteriatest(0,[u1,u2,u3])
                    if passed != 1:
                        # Returning nans or infs breaks the fits, so this was
                        # the best I could think of
                        light_curve = 1e12*np.ma.ones(time.shape)
                        continue
                    pl_params.u = np.array([u1, u2, u3])                    

                # Set spot parameters
                nspots = 0
                for key in self.keys:
                    split_key = key.split('_')
                    if len(split_key) == 1:
                        chan = 0
                    else:
                        chan = int(split_key[1])
                    if 'rad' in split_key[0]:
                        # Get the spot number and update self.spotrad
                        self.spotrad[chan, int(split_key[0][7:])] = \
                            np.array([self.parameters.dict[key][0]])
                        nspots += 1
                    elif 'spotu' in split_key[0]:
                        # For sampling on a unit disk
                        sampletype = 'unitdisk'
                        self.spotu[chan, int(split_key[0][5:])] = \
                            np.array([self.parameters.dict[key][0]])
                    elif 'spotv' in split_key[0]:
                        # For sampling on a unit disk
                        self.spotv[chan, int(split_key[0][5:])] = \
                            np.array([self.parameters.dict[key][0]])
                    elif 'con' in split_key[0]:
                        # Get the spot constrast and assign
                        spot_contrast = self.parameters.dict[key][0]
                    elif 'x' in split_key[0]:
                        # Get the spot x position
                        sampletype = 'xy'
                        self.spotx[chan, int(split_key[0][5:])] = \
                            np.array([self.parameters.dict[key][0]])
                    elif 'y' in split_key[0]:
                        # Get the spot y position
                        self.spoty[chan, int(split_key[0][5:])] = \
                            np.array([self.parameters.dict[key][0]])
                    elif 'lat' in split_key[0]:
                        sampletype = 'latlon'
                        # Get the spot lat and update self.spotlat
                        self.spotlat[chan, int(split_key[0][7:])] = \
                            np.array([self.parameters.dict[key][0]])
                    elif 'lon' in split_key[0]:
                        # Get the spot lon and update self.spotlon
                        self.spotlon[chan, int(split_key[0][7:])] = \
                            np.array([self.parameters.dict[key][0]])             
                    else:
                        # it's the number of points to evaluate
                        nrings = self.parameters.dict[key][0]
                
                # Sample in latitude and longitude OR in a unit disk and convert to x/y, assumed latitude is going from -90 to 90 degrees
                if (sampletype == 'latlon'):
                    tempx = np.cos(self.spotlon * np.pi/180) * np.sin((90-self.spotlat) * np.pi/180)
                    tempy = np.sin(self.spotlon * np.pi/180) * np.sin((90-self.spotlat) * np.pi/180)
                    # Need to shift to the coordinate frame of Fleck where x -> observer
                    self.spotx = tempy
                    self.spoty = np.sign(self.spotlat)*np.sqrt(1-tempx**2 - tempy**2)
                elif (sampletype == 'unitdisk'):
                    self.spotx = np.sqrt(self.spotu) * np.cos(2*np.pi*self.spotv)
                    self.spoty = np.sqrt(self.spotu) * np.sin(2*np.pi*self.spotv)
                # Weights: 2.0 times limb darkening times width of integration annulii.
                rrings = np.linspace(1.0/(2*nrings), 1.0-1.0/(2*nrings), nrings)
                weights = 2.0 * limbdarkening(rrings, pl_params.u, pl_params.limb_dark) / nrings
                # Get the X-Y coordinates and projected separation from center z, see Murray & Dermott 1998 or Winn 2010           
                trueanom = batman.TransitModel(pl_params, time, transittype='primary').get_true_anomaly()
                planetx, planety, planetz = planet_XYZ_position(trueanom,pl_params.a,pl_params.inc*np.pi/180,pl_params.ecc,pl_params.w*np.pi/180)
                planetangle = np.array([spotrod.circleangle(rrings, pl_params.rp, thisz) for thisz in planetz[(planetz < (1 + pl_params.rp))]])
                # Make the transit model
                modelstsp = np.ones(planetz.shape)
                modelstsp[planetz < (1 + pl_params.rp)] = spotrod.integratetransit(planetx[planetz < (1 + pl_params.rp)], #Planet x coordinate
                                                                                   planety[planetz < (1 + pl_params.rp)], #Planet y coordinate
                                                                                   planetz[planetz < (1 + pl_params.rp)], #Planet z coordinate
                                                                                   pl_params.rp, #rp/r*
                                                                                   rrings, #radii of integration annuli in stellar radii, non-decreasing
                                                                                   weights, #2.0 * limb darkening at r[i] * width of annulus i 
                                                                                   self.spotx[chan][:nspots], #X coording of spot
                                                                                   self.spoty[chan][:nspots], #Y coord of spot
                                                                                   self.spotrad[chan][:nspots], #Spot radius
                                                                                   np.tile(spot_contrast,nspots), #Spot contrast
                                                                                   planetangle)
                light_curve *= modelstsp

            lcfinal = np.ma.append(lcfinal, light_curve)

        return lcfinal


def correct_light_travel_time(time, pl_params):
    '''Correct for the finite light travel speed.

    This function uses the KeplerOrbit.py file from the Bell_EBM package
    as that code includes a newer, faster method of solving Kepler's equation
    based on Tommasini+2018.

    Parameters
    ----------
    time : ndarray
        The times at which observations were collected
    pl_params : batman.TransitParams or poet.TransitParams
        The TransitParams object that contains information on the orbit.

    Returns
    -------
    time : ndarray
        Updated times that can be put into batman transit and eclipse functions
        that will give the expected results assuming a finite light travel
        speed.

    Notes
    -----
    History:

    - 2022-03-31 Taylor J Bell
        Initial version based on the Bell_EMB KeplerOrbit.py file by
        Taylor J Bell and the light travel time calculations of SPIDERMAN's
        web.c file by Tom Louden
    '''
    # Need to convert from a/Rs to a in meters
    a = pl_params.a * (pl_params.Rs*const.R_sun.value)

    if pl_params.ecc > 0:
        # Need to solve Kepler's equation, so use the KeplerOrbit class
        # for rapid computation. In the SPIDERMAN notation z is the radial
        # coordinate, while for Bell_EBM the radial coordinate is x
        orbit = KeplerOrbit(a=a, Porb=pl_params.per, inc=pl_params.inc,
                            t0=pl_params.t0, e=pl_params.ecc, argp=pl_params.w)
        old_x, _, _ = orbit.xyz(time)
        transit_x, _, _ = orbit.xyz(pl_params.t0)
    else:
        # No need to solve Kepler's equation for circular orbits, so save
        # some computation time
        transit_x = a*np.sin(pl_params.inc)
        old_x = transit_x*np.cos(2*np.pi*(time-pl_params.t0)/pl_params.per)

    # Get the radial distance variations of the planet
    delta_x = transit_x - old_x

    # Compute for light travel time (and convert to days)
    delta_t = (delta_x/const.c.value)/(3600.*24.)

    # Subtract light travel time as a first-order correction
    # Batman will then calculate the model at a slightly earlier time
    return time-delta_t.flatten()

