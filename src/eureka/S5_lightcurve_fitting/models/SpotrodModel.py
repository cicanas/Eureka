import numpy as np
import astropy.units as unit
from ..limb_darkening_fit import ld_profile
from .AstroModel import true_anomaly
from astropy.coordinates import CartesianRepresentation, UnitSphericalRepresentation
from astropy.coordinates.matrix_utilities import rotation_matrix

try:
    import spotrod
except:
    pass

if not('spotrod' in locals()):
    try:
        from pyspotrod import pyspotrod as spotrod    
        print("Using C version of spotrod.")
    except:
        print("Using a slower python version of spotrod. Functionality may be limited.")
        from . import Slowspotrod as spotrod

from .BatmanModels import BatmanTransitModel


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
        """Initialize the spotrod model

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

        self.spotcon_file = kwargs.get('spotcon_file')
        if self.spotcon_file:
            # Load spot contrast coefficients from a custom file
            try:
                spot_coeffs = np.genfromtxt(self.spotcon_file)
            except FileNotFoundError:
                raise Exception(f"The spot contrast file {self.spotcon_file}"
                                " could not be found.")

            # Load all spot contrasts into the parameters object
            log = kwargs.get('log')
            log.writelog("Using the following spot contrast values:")
            for c in range(self.nchannel_fitted):
                chan = self.fitted_channels[c]
                if c == 0 or self.nchannel_fitted == 1:
                    chankey = ''
                else:
                    chankey = f'_ch{chan}'
                item = f'spotcon{chankey}'
                if item in self.paramtitles:
                    contrast_val = spot_coeffs[chan]
                    log.writelog(f"{item}: {contrast_val}")
                    # Use the file value as the starting guess
                    self.parameters.dict[item][0] = contrast_val
                    # In a normal prior, center at the file value
                    if (self.parameters.dict[item][-1] == 'N' and
                            self.recenter_spotcon_prior):
                        self.parameters.dict[item][-3] = contrast_val
                    # Update the non-dictionary form as well
                    setattr(self.parameters, item,
                            self.parameters.dict[item])


class TransitModel():
    """
    Class for generating model transit light curves with spotrod.
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
        spotx = np.zeros(0)
        spoty = np.zeros(0)
        spotcon = np.zeros(0)

        for n in range(pl_params.nspots):
            # read radii, latitudes, longitudes, and contrasts
            if n > 0:
                spot_id = f'{n}'
            else:
                spot_id = ''
            spotrad = np.hstack([spotrad, [getattr(pl_params, f'spotrad{spot_id}'),]])
            spotcon = np.concatenate([spotcon, [getattr(pl_params, f'spotcon{spot_id}'),]])
            if pl_params.samplingtype == 'latlon':
                spotlat = np.zeros(0)
                spotlon = np.zeros(0)
                spotlat = np.concatenate([spotlat, [getattr(pl_params, f'spotlat{spot_id}'),]])
                spotlon = np.concatenate([spotlon, [getattr(pl_params, f'spotlon{spot_id}'),]])
                # Convert latitude and longitude to X/Y, assuming no rotation of the star and zero obliquity
                # See fleck's function called spherical_to_cartesian
                # Represent those spots with cartesian coordinates (x, y, z)
                # In this coordinate system, the observer is at positive x->inf,
                # the star is at the origin, and (y, z) is the sky plane.
                stellar_inclination = rotation_matrix(pl_params.spotstari - 90*unit.deg, axis='y')
                cartesian = UnitSphericalRepresentation(spotlon*unit.degree, spotlat*unit.degree).represent_as(CartesianRepresentation).transform(stellar_inclination)              
                spotx = cartesian.y.value
                spoty = cartesian.z.value
            elif pl_params.samplingtype == 'xy':
                spotx = np.concatenate([spotx, [getattr(pl_params, f'spotx{spot_id}'),]])
                spoty = np.concatenate([spoty, [getattr(pl_params, f'spoty{spot_id}'),]])
            elif pl_params.samplingtype == 'unitdisk':
                spotu = np.atleast_1d(getattr(pl_params, f'spotu{spot_id}'))
                spotv = np.atleast_1d(getattr(pl_params, f'spotv{spot_id}'))
                # Convert U/V to X/Y
                spotx = np.concatenate([spotx, np.sqrt(spotu) * np.cos(2*np.pi*spotv)])
                spoty = np.concatenate([spoty, np.sqrt(spotu) * np.sin(2*np.pi*spotv)])
        if pl_params.spotnpts is None:
            # Have a default spotnpts for spotrod, this defines the number of rings for integration
            pl_params.spotnpts = 1000
        
        # Set nan contrasts to value of spot0
        tempcon = np.array(spotcon,copy=True)
        tempcon[np.isnan(spotcon)] = spotcon[0]

        inverse = False
        if pl_params.rp < 0:
            # The planet's radius is negative, so need to do some tricks to
            # avoid errors
            inverse = True
            pl_params.rp *= -1

        trueanom = true_anomaly(pl_params, self.t.data)
        # Weights: 2.0 times limb darkening times width of integration annulii.
        rrings = np.linspace(1.0/(2*pl_params.spotnpts), 1.0-1.0/(2*pl_params.spotnpts), pl_params.spotnpts)
        weights = 2.0 * limbdarkening(rrings, pl_params.u, pl_params.limb_dark) / pl_params.spotnpts
        # Get the X-Y coordinates and projected separation from center z, see Murray & Dermott 1998 or Winn 2010           
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
                                                                    spotx, #X coording of spot
                                                                    spoty, #Y coord of spot
                                                                    spotrad, #Spot radius
                                                                    tempcon, #Spot contrast
                                                                    planetangle)

        if inverse:
            # Invert the transit feature if rp<0
            lc = 2. - lc

        return lc
