import numpy as np
from ..lib import medstddev as msd


def sigrej(data, sigma, mask=None, estsig=None, ival=False, axis=0,
           fmean=False, fstddev=False, fmedian=False, fmedstddev=False):
    '''This function flags outlying points in a data set using sigma rejection.

    Parameters
    ----------
    data : ndarray
        Array of points to apply sigma rejection to.
    sigma : ndarray (1D)
        1D array of sigma values for each iteration of sigma rejection.
        Number of elements determines number of iterations.
    mask : byte array; optional
        Same shape as Data, where False indicates the corresponding element in
        Data is good and True indicates it is bad.  Only
        rejection of good-flagged data will be further
        considered.  This input mask is NOT modified in the caller. Defaults to
        None where only non-finite data is masked.
    estsig : ndarray; optional
        [nsig] array of estimated standard deviations to
        use instead of calculated ones in each iteration.
        This is useful in the case of small datasets with
        outliers, in which case the calculated standard
        deviation can be large if there is an outlier and
        small if there is not, leading to rejection of
        good elements in a clean dataset and acceptance of
        all elements in a dataset with one bad element.
        Set any element of estsig to a negative value to
        use the calculated standard deviation for that iteration.
    ival : ndarray (2D); optional
        (returned) 2D array giving the median and standard deviation (with
        respect to the median) at each iteration.
    axis : int; optional
        The axis along which to compute the mean/median.
    fmean : ndarray; optional
        (returned) the mean of the accepted data.
    fstddev : ndarray; optional
        (returned) the standard deviation of the accepted data with respect
        to the mean.
    fmedian : ndarray; optional
       (returned) the median of the accepted data.
    fmedstddev : ndarray; optional
       (returned) the standard deviation of the accepted data with respect
       to the median.

    Returns
    -------
    ret : tuple
        This function returns a mask of accepted values in the data.  The
        mask is a byte array of the same shape as Data.  In the mask, False
        indicates good data, True indicates an outlier in the corresponding
        location of Data. fmean, fstddev, fmedian, and fmedstddev will also
        be updated and returned if they were passed in. All of these will be
        packaged together into a tuple.

    Notes
    -----
    SIGREJ flags as outliers points a distance of sigma* the standard
    deviation from the median.  Unless given as a positive value in
    ESTSIG, standard deviation is calculated with respect to the
    median, using MEDSTDDEV. For each successive iteration and value of
    sigma SIGREJ recalculates the median and standard deviation from
    the set of 'good' (not masked) points, and uses these new values in
    calculating further outliers. The final mask contains a value of False
    for every 'inlier' and True for every outlying data point.
    '''
    # Get sizes
    dims = list(np.shape(data))
    nsig = np.size(sigma)
    if nsig == 0:
        nsig = 1
        sigma = [sigma]

    # Default mask: only non-finite values are bad
    if mask is None:
        mask = ~np.isfinite(data)

    # Apply the mask
    data = np.ma.masked_where(mask, data)

    # defining estsig makes the logic below easier
    # if estsig is None:
    #     estsig = - np.ones(nsig)

    # Return parameters:
    retival = ival
    retfmean = fmean
    retfstddev = fstddev
    retfmedian = fmedian
    retfmedstddev = fmedstddev

    # Remove axis
    del (dims[axis])
    ival = np.empty((2, nsig) + tuple(dims))
    ival[:] = np.nan

    # Iterations
    for iter in np.arange(nsig):
        if estsig is None:
            # note: ival is slicing
            ival[1, iter], ival[0, iter] = msd.medstddev(data, mask, axis=axis,
                                                         medi=True)
        # if estsig[iter] > 0:   # if we dont have an estimated std dev.
        else:
            # Calculations
            for j in np.arange(dims[0]):
                for i in np.arange(dims[1]):
                    ival[0, iter, j, i] = \
                        np.ma.median(data[:, j, i])
            # note: ival is slicing
            ival[1, iter] = estsig[iter]

            # Fixes
            count = np.sum(~mask, axis=axis)
            # note: ival is slicing
            (ival[1, iter])[np.where(count == 0)] = np.nan

        # Update mask
        # note: ival is slicing
        mask |= ((data < (ival[0, iter] - sigma[iter] * ival[1, iter])) |
                 (data > (ival[0, iter] + sigma[iter] * ival[1, iter])))

    # the return arrays
    ret = (mask,)
    if retival:
        ret = ret + (ival, )

    # final calculations
    if retfmean or retfstddev:
        count = np.sum(~mask, axis=axis)
        fmean = np.nansum(data*~mask, axis=axis)

        # calculate only where there are good pixels
        goodvals = np.isfinite(fmean) * (count > 0)
        if np.ndim(fmean) == 0 and goodvals:
            fmean /= count
        else:
            fmean[np.where(goodvals)] /= count[np.where(goodvals)]

        if retfstddev:
            resid = (data-fmean)*~mask
            fstddev = np.sqrt(np.sum(resid**2, axis=axis)/(count-1))
            if np.ndim(fstddev) == 0:
                if count == 1:
                    fstddev = 0.0
            else:
                fstddev[np.where(count == 1)] = 0.0

    if retfmedian or retfmedstddev:
        fmedstddev, fmedian = msd.medstddev(data, mask, axis=axis, medi=True)

    # the returned final arrays
    if retfmean:
        ret = ret + (fmean,)
    if retfstddev:
        ret = ret + (fstddev,)
    if retfmedian:
        ret = ret + (fmedian,)
    if retfmedstddev:
        ret = ret + (fmedstddev,)

    if len(ret) == 1:
        return ret[0]
    return ret
