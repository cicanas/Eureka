import numpy as np
import astropy.units as unit

try:
    from pyspotrod import pyspotrod as spotrod    
    print("Using C version of spotrod.")
except:
    print("Using a slower python version of spotrod. Functionality may be limited.")
    from . import Slowspotrod as spotrod

try:
    import os
    os.sys.path.append(os.environ['LDC3_PATH'])
    import LDC3
except:
    print("Could not import LDC3. Code will break if you specify an LD law of 'kipping2015'.")

from .BatmanModels import BatmanTransitModel

from .AstroModel import true_anomaly

from ..limb_darkening_fit import ld_profile


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


class SpotrodTransitModel(BatmanTransitModel):
    """Transit Model with Star Spots"""
    def __init__(self, **kwargs):
        """Initialize the fleck model

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
            Can pass in the parameters, longparamlist, nchan, and
            paramtitles arguments here.
        """
        # Inherit from BatmanTransitModel class
        super().__init__(**kwargs)
        self.name = 'spotrod transit'
        # Define transit model to be used
        self.transit_model = TransitModel


class TransitModel():
    """
    Class for generating model transit light curves with fleck.
    """
    def __init__(self, pl_params, t, transittype="primary"):
        """
        Does some initial pre-checks and saves some parameters.

        Parameters
        ----------
        pl_params : object
            Contains the physical parameters for the transit model.
        t : array
            Array of times.
        transittype : str; optional
            Options are primary or secondary.  Default is primary.
        """
        if transittype != "primary":
            raise ValueError('The spotrod transit model only allows transits and'
                             ' not eclipses.')

        # store t for later use
        self.t = t

    def light_curve(self, pl_params):
        """
        Calculate a model light curve.

        Parameters
        ----------
        pl_params : object
            Contains the physical parameters for the transit model.

        Returns
        -------
        lc : ndarray
            Light curve.
        """
        # create arrays to hold values
        spotrad = np.zeros(0)
        spotu = np.zeros(0)
        spotv = np.zeros(0)

        for n in range(pl_params.nspots):
            # read radii, latitudes, longitudes, and contrasts
            if n > 0:
                spot_id = f'{n}'
            else:
                spot_id = ''
            spotrad = np.concatenate([
                spotrad, [getattr(pl_params, f'spotrad{spot_id}'),]])
            spotu = np.concatenate([
                spotu, [getattr(pl_params, f'spotu{spot_id}'),]])
            spotv = np.concatenate([
                spotv, [getattr(pl_params, f'spotv{spot_id}'),]])

        spotx = np.sqrt(spotu) * np.cos(2*np.pi*spotv)
        spoty = np.sqrt(spotu) * np.sin(2*np.pi*spotv)

        if pl_params.spotnpts is None:
            # Have a default number of rings for spotrod
            pl_params.spotnpts = 1000

        inverse = False
        if pl_params.rp < 0:
            # The planet's radius is negative, so need to do some tricks to
            # avoid errors
            inverse = True
            pl_params.rp *= -1

        # Weights: 2.0 times limb darkening times width of integration annulii.
        rrings = np.linspace(1.0/(2*pl_params.spotnpts), 1.0-1.0/(2*pl_params.spotnpts), pl_params.spotnpts)
        weights = 2.0 * limbdarkening(rrings, pl_params.u, pl_params.limb_dark) / pl_params.spotnpts

        # Get the X-Y coordinates and projected separation from center z, see Murray & Dermott 1998 or Winn 2010
        trueanom = true_anomaly(pl_params, self.t.data)
        planetx, planety, planetz = planet_XYZ_position(trueanom,pl_params.a,pl_params.inc*np.pi/180,pl_params.ecc,pl_params.w*np.pi/180)
        planetangle = np.array([spotrod.circleangle(rrings, pl_params.rp, thisz) for thisz in planetz[(planetz < (1 + pl_params.rp))]])

        # Make the transit model
        lc = np.ones(planetz.shape)
        lc[planetz < (1 + pl_params.rp)] = spotrod.integratetransit(planetx[planetz < (1 + pl_params.rp)], #Planet x coordinate
                                                                    planety[planetz < (1 + pl_params.rp)], #Planet y coordinate
                                                                    planetz[planetz < (1 + pl_params.rp)], #Planet z coordinate
                                                                    pl_params.rp, #rp/r*
                                                                    rrings, #radii of integration annuli in stellar radii, non-decreasing
                                                                    weights, #2.0 * limb darkening at r[i] * width of annulus i 
                                                                    spotx[:, None], #X coording of spot
                                                                    spoty[:, None], #Y coord of spot
                                                                    spotrad[:, None], #Spot radius
                                                                    np.tile(pl_params.spotcon,pl_params.nspots), #Spot contrast
                                                                    planetangle)

        if inverse:
            # Invert the transit feature if rp<0
            lc = 2. - lc

        return lc
