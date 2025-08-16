import numpy as np

def circleangle(r, p, z):
    """circleanglesorted(r, p, z)
  Calculate half central angle of the arc of circle of radius r
  (which concentrically spans the inside of the star during integration)
  that is inside a circle of radius p (planet)
  with separation of centers z.
  This is a zeroth order homogeneous function, that is,
  circleangle(alpha*r, alpha*p, alpha*z) = circleangle(r, p, z).

  This version uses a binary search on the sorted r.

  Input:
    r  one dimensional numpy array, must be increasing
    p  scalar
    z  scalar
  They should all be non-negative, but there is no other restriction.

  Output:
    circleangle  one dimensional numpy array, same size as r
  """
    # If the circle arc of radius r is disjoint from the circular disk 
    # of radius p, then the angle is zero.
    answer = np.empty_like(r)
    if (p > z):
        # Planet covers center of star.
        a, b = np.searchsorted(r, [p-z, p+z], side="right")
        answer[:a] = np.pi
        answer[a:b] = np.arccos((r[a:b]*r[a:b]+z*z-p*p)/(2*z*r[a:b]))
        answer[b:] = 0.0
    else:
        # Planet does not cover center of star.
        a, b = np.searchsorted(r, [z-p, z+p], side="right")
        answer[:a] = 0.0
        answer[a:b] = np.arccos((r[a:b]*r[a:b]+z*z-p*p)/(2*z*r[a:b]))
        answer[b:] = 0.0
    return answer


def ellipseangle(r, a, z):
    '''Calculate half central angle of the arc of circle of radius r
  (which concentrically spans the inside of the star during integration)
  that is inside an ellipse of semi-major axis a with separation of centers z.
  The orientation of the ellipse is so that the center of the circle lies on 
  the continuation of the minor axis. This is the orientation if the ellipse
  is a circle on the surface of a sphere viewed in projection, and the circle
  is concentric with the projection of the sphere.
  b is calculated from a and z, assuming projection of a circle of radius a
  on the surface of a unit sphere. If a and z are not compatible, a is clipped.
  This is not zeroth order homogeneous function, because it calculates b based
  on a circle of radius a living on the surface of the unit sphere.
  r is an array, a, and z are scalars. They should all be non-negative.
  We store the result on the n double positions starting with *answer.
  
  Input:

  r        radius of circle [n]
  a        semi-major axis of ellipse, non-negative
  z        distance between centers of circle and ellipse,
           non-negative and at most 1
  n        size of array a

  Output:

  answer   half central angle of arc of circle that lies inside ellipes [n].'''
    answer = np.zeros(len(r))
    ## Degenerate case
    if (a <= 0) or (z >= 1):
        pass
    ## Concentric case.
    elif (z<=0):
        # answer[r<a] = np.pi
        bound = np.searchsorted(r, a, side="right")
        answer[:bound] = np.pi
    ## Unphysical case: clip a. Now b=0.
    elif (a**2 + z**2) >= 1.0:
        bound = np.searchsorted(r, z, side="right")
        answer[bound:] = np.arccos(z/r[:bound])
    ## Case of ellipse.
    else:
        ## Equation A3
        b = a * np.sqrt(1-z**2/(1-a**2))
        ## Calculate A based on a and z to mitigate rounding errors.
        A = z**2 / (1.0 - a**2 - z**2)
        ## First, go through all the cases where the ellipse covers C_r.
        ## If there is no such r, then bound=0, nothing happens.
        bound1,bound2 = np.searchsorted(r, [b-z,z-b], side="right")
        answer[:bound1] = np.pi
        ## Now go through all the cases when C_r does not reach out to the ellipse.
        ## Again, the loop body might not get executed. ##
        # answer[bound1:bound2] = 0
        ## Now take care of the cases where C_r and the ellipse intersect. z_crit = 1-a^2.
        if (z < 1.0 - a**2):
            bound3 = np.searchsorted(r, z+b, side="right")
        else:
            bound3 = len(r)
        ## If not disjoint from the outside.
        ## We must have y_+ > -b now.
        answer[bound2.clip(bound1):bound3] = np.arccos((z - (-z + np.sqrt(z**2 - A*(r[bound2.clip(bound1):bound3]**2 - z**2 - a**2)))/A)/r[bound2.clip(bound1):bound3])
    return answer


def integratetransit(planetx, planety, z, p, r, f, spotx, spoty, spotradius, spotcontrast, planetangle):
    '''Calculate integrated flux of a star if it is transited by a planet
    of radius p*R_star, at projected position (planetx, planety)
    in R_star units.
    Flux is normalized to out-of-transit flux.
    This algorithm works by integrating over concentric rings,
    the number of which is controlled by n. Use n=1000 for fair results.
    Planetx is the coordinate perpendicular to the transit chord
    normalized to stellar radius units, and planety is the one
    parallel to the transit chord, in a fashion such that it increases
    throughout the transit.
    We assume that the one-dimensional arrays spotx, spoty, spotradius
    and spotcontrast have the same length: the number of the spots.

    Input parameters:

    planet[xy]    planetary center coordinates in stellar radii in sky-projected coordinate system [m]
    z             planetary center distance from stellar disk center in stellar radii     (cached) [m]
    p             planetary radius in stellar radii, scalar
    r             radii of integration annuli in stellar radii, non-decreasing            (cached) [n]
    f             2.0 * limb darkening at r[i] * width of annulus i                       (cached) [n]
    spotx, spoty  spot center coordinates in stellar radii in sky-projected coordinate system      [k]
    spotradius    spot radius in stellar radii                                                     [k]
    spotcontrast  spot contrast                                                                    [k]
    planetangle   value of [circleangle(r, p, z[i]) for i in range(m)], delta(r)          (cached) [m,n]

    (cached) means the parameter is redundant, and could be calculated from other parameters,
    but storing it and passing it to this routine speeds up iterative execution (fit or MCMC).
    Note that we do not take limb darkening coefficients, all we need is f.

    Output parameters:

    answer        model lightcurve, with oot=1.0
    '''
    ### If we have no spot:
    answer = np.zeros(len(z))
    intransit = (z < 1 + p)
    answer[~intransit] = 1
    if len(spotx) == 0:
        # ## Evaluate the integral for ootflux using the trapezoid method.
        ootflux = np.sum(np.pi*r*f)
        answer[intransit] = np.sum(r * f * (np.pi - planetangle[intransit,:]),axis=1) / ootflux
    else:
        ## Equation A2
        spotcenterdistance = np.sqrt( (spotx**2 + spoty**2) * (1.0 - spotradius**2) )
        spotangle = np.zeros((len(spotx),len(r)))
        ## Loop over spots: fill up some arrays that do not depend on z.
        for K in range(0,len(spotx)):
            # Calculate the half central angles of the spot, and store it in a single row of the 2D array.
            # These values do not depend on z, that's why we cache them for all spots.
            # Calculate gamma(r)
            spotangle[K] = ellipseangle(r, spotradius[K], spotcenterdistance[K])
        ## Evaluate the integral for ootflux using the trapezoid method.
        ## Equation 3 in manuscript
        ootflux = np.sum((np.pi + np.nansum(np.atleast_2d((spotcontrast - 1)).T * spotangle,axis=0))*f*r)
        ## Loop over observation times and calculate answer.
        ## Equation 4 in the manuscript
        if intransit.sum():
            ## Equation 4: pi - delta(r)
            values = np.pi - planetangle[intransit,:]
            ## Cycle through spots and add their contributions
            ## Equation 4: Sum( (f-1)*gammastar(r) )
            for K in range(0,len(spotx)):
                ## Calculate distance of spot center and planet center for these times
                d = np.sqrt( ( planetx - spotx[K] * np.sqrt(1.0 - spotradius[K]**2) )**2 + ( planety - spoty[K] * np.sqrt(1.0 - spotradius[K]**2) )**2)
                ## Calculate central angle between planet and spot, aka theta, as a function of time
                planetspotangle = np.zeros(len(z))
                if (spotcenterdistance[K] != 0):
                    planetspotangle[z!=0] = np.arccos( (z[z!=0]**2 + spotcenterdistance[K]**2 - d[z!=0]**2) / (2.0 * z[z!=0] * spotcenterdistance[K]) )
                ## Cycle through annuli. Equation A6
                ## Case 1: planet and spot arcs are disjoint, contributions add up.
                case1 = np.atleast_2d(planetspotangle).T > (spotangle[K] + planetangle[intransit,:])
                if case1.sum():
                    values[case1] +=  (((spotcontrast[K]-1.0) * spotangle[K])*case1.astype(float))[case1]
                ## Case 2: planet arc inside spot arc.
                case2 = (spotangle[K] > (planetangle[intransit,:] + np.atleast_2d(planetspotangle).T))
                if case2.sum():
                    values[case2] += ((spotcontrast[K]-1.0) * (spotangle[K] - planetangle[intransit,:]))[case2]
                ## Case 4: triangle inequality holds, partial overlap.         
                case3 = (planetangle[intransit,:] <= (spotangle[K] + np.atleast_2d(planetspotangle).T)) & np.logical_not(case1 | case2)
                if case3.sum():
                    ## Case 4a: partial overlap on one side only.
                    subcase = ((2*np.pi) >= (spotangle[K] + planetangle[intransit,:] + np.atleast_2d(planetspotangle).T)) & case3
                    if subcase.sum():
                        values[subcase] += (0.5 * (spotcontrast[K]-1.0) * (spotangle[K] + np.atleast_2d(planetspotangle).T - planetangle[intransit,:]))[subcase]                    
                    subcase = ((2*np.pi) < (spotangle[K] + planetangle[intransit,:] + np.atleast_2d(planetspotangle).T)) & case3                 
                    ## Case 4b: partial overlap on two sides.
                    if subcase.sum():
                        values[subcase] += ((spotcontrast[K]-1.0) * (np.pi - planetangle[intransit,:]))[subcase]
            ## Now we multiply the half arc length integrated contrast by r*f
            ## to get the integrand, and sum it up right away. */
            answer[intransit] = np.sum(values*f*r,axis=1) / ootflux
    return answer

