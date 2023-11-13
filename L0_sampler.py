import torch
import numpy as np
import math


def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    '''
    This is the official implementation of NeRF's HVS. It is also the most commonly used method when compared with ours.
    '''

    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def map_poly(coeff, xx, monic=False, deri=False):
    ''' 
    Find the values of a group of polynomials {p(x)} at position {x}, 
    i.e. : {p1(x), p2(x), ...} + {x1, x2, ...} -> {p1(x1), p2(x2), ...}
    Args:
        coeff: [batchsize, degree + 1]: the coefficient of polynomials {p(x)}. Indexes of coefficients and degrees are matched.
        xx: [batchsize]: the position x for each p(x).
        monic: True or False: if each p(x) is monic. if so, the coeff presents the rest coefficients
        deri: True or False: if deri = True, return the derivatives. i.e. {p1'(x1), p2'(x2), ...}
    Return:
        [batchsize]: the values at each points
    '''

    xx_new = xx.unsqueeze(-1)

    degree = coeff.size()[-1] - 1
    yy = 0

    for i in range(0, degree + 1):
        if deri:
            if i > 0:
                yy = yy + i * coeff[..., i].unsqueeze(-1) * xx_new**(i - 1)
        else: 
            yy = yy + coeff[..., i].unsqueeze(-1) * xx_new**i # Indexes of coefficients and degrees are matched.
    if monic:
        if deri:
            yy = yy + (degree + 1) * xx_new**degree
        else:
            yy = yy + xx_new**(degree + 1)
    return yy.squeeze(-1)

def get_integral_poly(coeff, only_coeff = False):
    ''' 
    Find the integral of a group of polynomials {p(x)} at position {x}, 
    i.e. : {p1(x), p2(x), ...} -> {\int p1(x), \int p2(x), ...}
    Args:
        coeff: [batchsize, degree + 1]: the coefficent of polynomials {p(x)}. Indexes of coefficients and degrees are matched.
        only_coeff: True or False: if True, return the coefficients of integral of {p(x)}; otherwise, return their definite integral values from 0 to 1.
    Return: coeff1 or integ
        coeff1: [batchsize, degree + 2]: the coefficients of integral of {p(x)}
        integ: [batchsize]: their definite integral values from 0 to 1
    '''
    degree = coeff.size()[-1]
    if only_coeff:
        coeff1 = torch.cat([torch.zeros_like(coeff[..., :1]), coeff],dim = -1)
        for i in range(2, degree + 1):
            coeff1[..., i] = coeff1[..., i] / i
        return coeff1
    else:
        integ = coeff[..., 0]
        for i in range(1, degree):
            integ = integ + coeff[..., i] / (i + 1)

    return integ

def find_roots_bi(coeff, epsilon = 1e-4, monic = False):
    ''' 
    Use binary search to find roots of a group of polynomials {p(x)} in [0, 1]. 
    As it is used in inverse transform sampling, the roots exist theoretically. 
    i.e. : {p1(x), p2(x), ...} -> {x1, x2, ...}

    (We also tried methods like Newton method, but it seems that binary search is enough for using here, and its stability is really good.)
    Args:
        coeff: [batchsize, degree + 1]: the coefficent of polynomials {p(x)}. Indexes of coefficients and degrees are matched.
        epsilon: a number: when p(x) < epsilon, x can be considered as the root
        monic: True or False: if each p(x) is monic. if so, the coeff presents the rest coefficients
    Return: 
        roots_mid: the roots found in [0, 1]
    '''
    roots_below = torch.zeros_like(coeff[..., 0])
    roots_above = torch.ones_like(coeff[..., 0])
    roots_mid = 0.5 * torch.ones_like(coeff[..., 0])
    
    fx = torch.ones_like(coeff[..., 0])
    n = 0 # Number of cycles, can be changed
    while torch.max(torch.abs(fx)) > epsilon and n < 10:
        roots_mid = 0.5 * (roots_below + roots_above)
        fx = map_poly(coeff, roots_mid, monic = monic)
        mask = fx < 0
        roots_below[mask] = roots_mid[mask]
        roots_above[~mask] = roots_mid[~mask]

        n = n + 1
    roots_mid = 0.5 * (roots_below + roots_above)

    return roots_mid

'''
The classes below are different functions for interpolation. Each of them contains a function get_integral(...)
for normalization, and a function find_roots(...) for inverse transform sampling.

Some of them may have another parameters. Their definations are written in the classes.
'''

class LinearInterpolation(object):
    ''' 
    Use piecewise linear function to interpolate in each interval

    Definition: Suppose w(0)=a, w(1)=b, then the function is:
        w(t) = a + (b-a)t

    See the paper for more details
    '''
    @staticmethod
    def get_integral(yy, parameter=None):
        '''
        Find the definite integral value of w(t) from 0 to 1
        Args:
            yy: [batchsize, n_sample]: the coarse {wi} on each ray
        Return: 
            integ: [batchsize, n_sample-1]: the definite integral value of w(t) from 0 to 1 in each interval
        '''
        integ = (yy[..., 1:] + yy[..., :-1]) / 2

        return integ
    
    @staticmethod
    def find_roots(weights, rhs, parameter=None):
        '''
        Find the roots needed in inverse transform sampling.
        Args:
            weights: [batchsize, 2]: a and b, i.e. values at the endpoints of each interval
            rhs: [batchsize]: the residual probabilities of each interval
        Return: 
            roots: [batchsize]: the sample position in each interval
        '''
        bb = weights[..., 0]
        aa = weights[..., 1] - weights[..., 0] + 1e-7
        delta = torch.sqrt(bb ** 2 + 2*aa*rhs)
        # roots = (-bb + delta) / aa # this is the direct solution
        roots = 2 * rhs / (bb + delta) # this is the stable solution

        return roots
    
class ExponentialInterpolation(object):
    ''' 
    Use piecewise exponential function to interpolate in each interval. It might have the best performance.

    Definition: Suppose w(0)=a, w(1)=b, then the function is:
        w(t) = a (b/a)^t

    See the paper for more details
    '''
    @staticmethod
    def get_integral(yy, parameter=None):
        '''
        Find the definite integral value of w(t) from 0 to 1
        Args:
            yy: [batchsize, n_sample]: the coarse {wi} on each ray
        Return: 
            integ: [batchsize, n_sample-1]: the definite integral value of w(t) from 0 to 1 in each interval
        '''
        integ = (yy[..., 1:] - yy[..., :-1])/(torch.log(yy[..., 1:]/yy[..., :-1]) + 1e-6)

        return integ
    
    @staticmethod
    def find_roots(weights, rhs, parameter=None):
        '''
        Find the roots needed in inverse transform sampling.
        Args:
            weights: [batchsize, 2]: a and b, i.e. values at the endpoints of each interval
            rhs: [batchsize]: the residual probabilities of each interval
        Return: 
            roots: [batchsize]: the sample position in each interval
        '''
        denom = torch.log(weights[..., 1]/weights[..., 0]) + 1e-6
        roots = torch.log1p(rhs*denom/weights[..., 0]) / denom

        return roots
    
class InverseInterpolation(object):
    ''' 
    Use piecewise inverse function to interpolate in each interval. It also have good performance.

    Definition: Suppose w(0)=a, w(1)=b, then the function is: (a little complex)
        w(t) = a/(((a/b)^(1/p)-1)t + 1)^p
        where p is a parameter
        when p=1, w(t) = ab/((a-b)t + b), which is used the most

    See the paper for more details.
    '''
    @staticmethod
    def get_integral(yy, parameter = 1.):
        '''
        Find the definite integral value of w(t) from 0 to 1
        Args:
            yy: [batchsize, n_sample]: the coarse {wi} on each ray
            parameter: a number: a parameter to adjust the shape of w(t). See the Definition above.
        Return: 
            integ: [batchsize, n_sample-1]: the definite integral value of w(t) from 0 to 1 in each interval
        '''
        if parameter < 1.+1e-4:
            integ = yy[..., 1:]*yy[..., :-1]*(torch.log(yy[..., 1:]/yy[..., :-1]))/(yy[..., 1:] - yy[..., :-1]+1e-6)
        else:
            integ = yy[..., :-1]*(torch.pow(yy[..., 1:]/yy[..., :-1], 1/parameter-1)-1)/(1-parameter)/(torch.pow(yy[..., 1:]/yy[..., :-1], 1/parameter)-1+1e-8)

        return integ
    
    @staticmethod
    def find_roots(weights, rhs, parameter = 1.):
        '''
        Find the roots needed in inverse transform sampling.
        Args:
            weights: [batchsize, 2]: a and b, i.e. values at the endpoints of each interval
            rhs: [batchsize]: the residual probabilities of each interval
            parameter: a number: a parameter to adjust the shape of w(t). See the Definition above.
        Return: 
            roots: [batchsize]: the sample position in each interval
        '''
        aa = weights[..., 0]
        bb = weights[..., 1]
        if parameter < 1.+1e-4:
            roots = bb*(torch.exp(rhs*(aa-bb)/(aa*bb))-1)/(aa-bb+1e-6)
        else:
            denom = torch.pow(aa/bb, 1/parameter)-1
            roots = (torch.pow(rhs*(1-parameter)/aa*denom+1, 1/(1-parameter))-1)/(denom+1e-8)
        
        return roots

class PiecePolyInterpolation(object):
    ''' 
    Use piecewise polynomials to interpolate in each interval.

    Definition: Suppose w(0)=a, w(1)=b, then the function is: 
        w(t) = a + (b-a)t^p
        where p is a parameter
        when p=1, w(t) = a + (b-a)t, which is the piecewise linear function above
    '''
    @staticmethod
    def get_integral(yy, parameter = 1.):
        '''
        Find the definite integral value of w(t) from 0 to 1
        Args:
            yy: [batchsize, n_sample]: the coarse {wi} on each ray
            parameter: a number: a parameter to adjust the shape of w(t). See the Definition above.
        Return: 
            integ: [batchsize, n_sample-1]: the definite integral value of w(t) from 0 to 1 in each interval
        '''
        symmetry = True # if True, a<b and a>b have different expression to keep the shape
        coeff_0 = torch.zeros_like(yy)
        coeff_k = torch.zeros_like(yy)
        if symmetry:
            coeff_0 = torch.min(yy[..., 1:], yy[..., :-1])
            coeff_k = torch.abs(yy[..., 1:] - yy[..., :-1])
        else:
            coeff_0 = yy[..., :-1]
            coeff_k = yy[..., 1:] - yy[..., :-1]
        # coeff = torch.stack([coeff_0, coeff_k], dim = -1)

        integ = coeff_0 + coeff_k / (parameter + 1)

        return integ
    
    @staticmethod
    def find_roots(weights, rhs, parameter = 1., epsilon = 1e-5):
        '''
        Find the roots needed in inverse transform sampling.
        Args:
            weights: [batchsize, 2]: a and b, i.e. values at the endpoints of each interval
            rhs: [batchsize]: the residual probabilities of each interval
            parameter: a number: a parameter to adjust the shape of w(t). See the Definition above.
            epsilon: a number: used in binary search. When p(x) < epsilon, x can be considered as the root.
        Return: 
            roots: [batchsize]: the sample position in each interval
        '''
        symmetry = True
        if symmetry:
            coeff_0 = torch.min(weights[..., 1], weights[..., 0])
            coeff_k = torch.abs(weights[..., 1] - weights[..., 0])
        else:
            coeff_0 = weights[..., 0]
            coeff_k = weights[..., 1] - weights[..., 0]

        roots_below = torch.zeros_like(coeff_0)
        roots_above = torch.ones_like(coeff_0)
        roots_mid = 0.5 * torch.ones_like(coeff_0)
        
        fx = torch.ones_like(coeff_0)
        n = 0
        while torch.max(torch.abs(fx)) > epsilon and n < 11:
            roots_mid = 0.5 * (roots_below + roots_above)
            fx = weights[..., 0] * roots_mid + (weights[..., 1] - weights[..., 0]) * torch.pow(roots_mid, parameter+1)/(parameter+1) - rhs
            if symmetry:
                mask_sym = weights[..., 1] > weights[..., 0]
                fx[mask_sym] = (weights[..., 1] * roots_mid + (weights[..., 1] - weights[..., 0]) * (torch.pow(1-roots_mid, parameter+1) - 1)/(parameter+1) - rhs)[mask_sym]

            mask = fx < 0
            roots_below[mask] = roots_mid[mask]
            roots_above[~mask] = roots_mid[~mask]

            n = n + 1
        roots_mid = 0.5 * (roots_below + roots_above)

        return roots_mid

class TrapezoidInterpolation(object):
    ''' 
    Use the combination of linear and constant functions to interpolate in each interval.

    Definition: Suppose w(0)=a, w(1)=b, then the function is: 

                | a, if 0<= t <=p
        w(t) = -| ((b-a)t + a - p(a+b))/(1-2p), if p<= t <=1-p
                | b, if 1-p<= t <=1

        where p is a parameter.
        When p=0, w(t) = a + (b-a)t, which is the piecewise linear function above.
        When p=0.5, w(t) is piecewise constant, which is the original HVS.
    '''
    @staticmethod
    def get_integral(yy, parameter=None):
        '''
        Find the definite integral value of w(t) from 0 to 1. Note that it is not related with parameter.
        Args:
            yy: [batchsize, n_sample]: the coarse {wi} on each ray
        Return: 
            integ: [batchsize, n_sample-1]: the definite integral value of w(t) from 0 to 1 in each interval
        '''
        integ = (yy[..., 1:] + yy[..., :-1]) / 2 + 1e-8

        return integ
    
    @staticmethod
    def find_roots(weights, rhs, parameter = 0.):
        '''
        Find the roots needed in inverse transform sampling.
        Args:
            weights: [batchsize, 2]: a and b, i.e. values at the endpoints of each interval
            rhs: [batchsize]: the residual probabilities of each interval
            parameter: a number: a parameter to adjust the shape of w(t). See the Definition above.
        Return: 
            roots: [batchsize]: the sample position in each interval
        '''
        mask1 = rhs < weights[..., 0] * parameter
        mask3 = rhs > weights[..., 0] * 0.5 + weights[..., 1] * (1-2*parameter)/2

        aa = (weights[..., 1] - weights[..., 0])/(2-4*parameter)
        bb = (weights[..., 0] - parameter*(weights[..., 1] + weights[..., 0]))/(1-2*parameter)
        cc = aa*parameter*parameter + bb*parameter + rhs-weights[..., 0]*parameter
        delta = torch.sqrt(bb ** 2 + 4*aa*cc)
        # roots = (-bb + delta) / aa
        roots = 2 * rhs / (bb + delta + 1e-8)

        roots[mask1] = (rhs/weights[..., 0])[mask1]
        roots[mask3] = ((rhs - (weights[..., 0] * 0.5 + weights[..., 1] * (1-2*parameter)/2))/weights[..., 1] + 1 - parameter)[mask3]

        return roots

class CubicInterpolation(object):
    ''' 
    Use the cubic spline to interpolate in each interval.
    '''
    @staticmethod
    def get_polynomial(yy):
        '''
        Calculate the coefficients of cubic splines.
        Args:
            yy: [batchsize, n_sample]: the coarse {wi} on each ray
        Return: 
            coeff: [batchsize, n_sample-1, 4]: the coefficients of cubic splines of each interval on each ray
            degree: a number: the degree of splines
        '''
        degree = 3
        bins = torch.linspace(0, yy.shape[-1]-1, yy.shape[-1], device=yy.device).expand(yy.shape)
        batch_size = bins.size()[0]
        pts_num = bins.size()[1]
        bin_length = bins[..., 1:] - bins[..., :-1] # [b, n]
        matrix = torch.diag_embed(4 * torch.ones_like(bin_length[..., :-1]))
        for i in range(pts_num - 3):
            matrix[..., i, i + 1] = 1
            matrix[..., i + 1, i] = 1
        bi = 6 * (yy[..., 1:] - yy[..., :-1]).contiguous() # [b, n]
        vi = bi[..., 1:] - bi[..., :-1]
        second_dev = torch.linalg.solve(matrix, vi)# [b, n-1]
        second_dev = torch.cat([torch.zeros_like(second_dev[..., :1]), second_dev, torch.zeros_like(second_dev[..., :1])], dim = -1)# [b, n+1]

        coeff = torch.zeros([batch_size, pts_num - 1, 4], device=bins.device)
        coeff[..., 3] = (second_dev[..., 1:] - second_dev[..., :-1]) / (6 * bin_length)
        coeff[..., 2] = second_dev[..., :-1] / 2
        coeff[..., 1] = -bin_length*second_dev[..., 1:]/6 - bin_length*second_dev[..., :-1]/3 + (yy[..., 1:] - yy[..., :-1]) / bin_length
        coeff[..., 0] = yy[..., :-1]
    
        return coeff, degree
    
    @staticmethod
    def get_integral(coeff, only_coeff):
        '''
        Find the definite integral value of w(t) from 0 to 1. 
        See the function "get_integral_poly" for details.
        '''
        integ= get_integral_poly(coeff, only_coeff=only_coeff)

        return integ
    
    @staticmethod
    def find_roots(int_coeff):
        '''
        Find the roots needed in inverse transform sampling. 
        See the function "find_roots_bi" for details.
        '''
        roots = find_roots_bi(int_coeff, epsilon = 1e-4)

        return roots

class AkimaInterpolation(object):
    ''' 
    Use the Akima spline to interpolate in each interval. Definition can be found in https://www.mathworks.com/help/matlab/ref/makima.html
    '''
    @staticmethod
    def get_polynomial(yy):
        '''
        Calculate the coefficients of Akima splines.
        Args:
            yy: [batchsize, n_sample]: the coarse {wi} on each ray
        Return: 
            coeff: [batchsize, n_sample-1, 4]: the coefficients of Akima splines of each interval on each ray
            degree: a number: the degree of Akima spline
        '''
        degree = 3
        batch_size = yy.shape[0]
        pts_num = yy.shape[1]
        kk = (yy[..., 1:] - yy[..., :-1])
        kk = torch.cat([torch.zeros_like(kk[..., :2]), kk, torch.zeros_like(kk[..., :2])], dim = -1)
        w1 = torch.abs(kk[..., 3:] - kk[..., 2:-1]) +0.5*torch.abs(kk[..., 3:] + kk[..., 2:-1]) +1e-5
        w2 = torch.abs(kk[..., 1:-2] - kk[..., :-3]) +0.5*torch.abs(kk[..., 1:-2] + kk[..., :-3]) +1e-5
        dd = (w1 * kk[..., 1:-2] + w2 * kk[..., 2:-1]) / (w1 + w2)
        
        d1 = dd[..., :-1]
        d2 = dd[..., 1:]
        y1 = yy[..., :-1]
        y2 = yy[..., 1:]

        coeff = torch.zeros([batch_size, pts_num - 1, 4], device=yy.device)
        coeff[..., 3] = 2*y1 - 2*y2 + d1 + d2
        coeff[..., 2] = -3*y1 + 3*y2 - 2*d1 - d2
        coeff[..., 1] = d1
        coeff[..., 0] = y1
    
        return coeff, degree
    
    @staticmethod
    def get_integral(coeff, only_coeff):
        '''
        Find the definite integral value of w(t) from 0 to 1. 
        See the function "get_integral_poly" for details.
        '''
        integ= get_integral_poly(coeff, only_coeff=only_coeff)

        return integ
    
    @staticmethod
    def find_roots(int_coeff):
        '''
        Find the roots needed in inverse transform sampling. 
        See the function "find_roots_bi" for details.
        '''
        roots = find_roots_bi(int_coeff, epsilon = 1e-4)

        return roots

spline_dict = {'linear': LinearInterpolation, 'exp': ExponentialInterpolation,
               'inv': InverseInterpolation, 'piecepoly': PiecePolyInterpolation, 
               'tpz': TrapezoidInterpolation, 'cubic': CubicInterpolation, 
               'akima': AkimaInterpolation}


#####################
# The main function #
#####################

def L0_sample_pdf(bins, weights, N_samples, spline_type = 'exp', det=True, blur = True, ddblur=False, parameter = 0.):
    '''
    Our main function. 
    Args:
        bins: [batchsize, n_sample_coarse]: the coarse {t_i} sampled uniformly on each ray
        weights: [batchsize, n_sample_coarse]: the coarse {w_i} on each ray
        spline_type: string: the type of function chosen to interpolate. It should in spline_dict.
        N_samples: a number: the number of sampling points in fine stage
        det: True or False: choose different strategy when doing inverse transform sampling
        blur: True or False: if True, use the Maxblur strategy
        ddblur: True or False: if True and blur=True, use the blur strategy proposed in DDNeRF
        parameter: a number: in case if the function we choose need a parameter
    Return: 
        samples: [batchsize, N_samples]: sampling results on each ray in fine stage
    '''
    # Maxblur
    if blur:
        weights_pad = torch.cat([
            weights[..., :1],
            weights,
            weights[..., -1:],
        ],
            axis=-1)

        weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
        weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

        if not ddblur:
            weights = weights_blur + 0.001

        else: # The method proposed in DDNeRF
            prev = weights_pad[..., :-2]
            next = weights_pad[..., 2:]

            weights = 0.8*weights + 0.1*prev + 0.1*next + 0.01

    weights = weights + 1e-7

    # Get the interpolation class:
    function_class = spline_dict[spline_type]

    # Get definite integral of each interval. Methods using polynomials are treated differently.
    coeff = None
    degree = None
    special_poly = spline_type in ['akima', 'cubic']
    if special_poly:
        coeff, degree = function_class.get_polynomial(weights)
        integral= function_class.get_integral(coeff, only_coeff=False)
    else:
        integral= function_class.get_integral(weights, parameter)

    integral = torch.max(1e-7*torch.ones_like(integral), integral)
    
    # Get PDF and CDF
    pdf = integral / torch.sum(integral, dim=-1, keepdim=True)
    
    cdf = torch.cumsum(pdf[..., :-1], -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf, torch.ones_like(cdf[...,:1])], -1)  # (batch, len(bins_change)) # 62

    # Uniform sampling y
    if det:
        u = torch.linspace(
            0.0, 1.0, steps=N_samples, dtype=weights.dtype, device=weights.device
        )
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else: # borrowed from Mip-NeRF
        s = 1 / N_samples
        u = (torch.arange(N_samples, device=weights.device) * s).expand(list(cdf.shape[:-1]) + [N_samples])
        u = u + (torch.rand(cdf.shape[0], N_samples, device=weights.device)/((1/s) + 1e-5))
        u = torch.minimum(u, torch.tensor(0.9999))


    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)

    residual = u-cdf_g[...,0]

    # find the roots in each interval
    if special_poly:
        pre_int_coeff = function_class.get_integral(coeff, only_coeff=True)
        inds_g2 = torch.min((pre_int_coeff.shape[-2]-1) * torch.ones_like(below), below).unsqueeze(-1).expand(list(inds_g.shape[:-1])+[degree + 2])
        int_coeff = torch.gather(pre_int_coeff, 1, inds_g2)
        int_coeff[..., 0] = - residual * torch.sum(integral, dim=-1, keepdim=True)

        # the 2 lines below are for make the coefficients' range smaller to alleviate roundoff errors, but it seems that it is not necessary.
        # metric = torch.min(torch.abs(int_coeff), dim = -1)[0].unsqueeze(-1) + 1e-7
        # int_coeff = int_coeff / metric

        roots = find_roots_bi(int_coeff)
        
    else:
        rhs = residual * torch.sum(integral, dim = -1, keepdim = True)
        weights_g = torch.gather(weights.unsqueeze(1).expand(matched_shape), 2, inds_g)
        roots = function_class.find_roots(weights_g, rhs, parameter)

    t = torch.clip(torch.nan_to_num((roots), 0), 0, 1)
    samples = bins_g[...,0] + t*(bins_g[...,1]-bins_g[...,0])

    return samples

def max_sample_pdf(bins, weights, N_samples, blur = False, ddblur=False):
    '''
    Directly sample around the maximum {wi}. Results are always poor, just designed for comparison.
    Args:
        bins: [batchsize, n_sample_coarse]: the coarse {t_i} sampled uniformly on each ray
        weights: [batchsize, n_sample_coarse]: the coarse {w_i} on each ray
        N_samples: a number: the number of sampling points in fine stage
        blur: True or False: if True, use the Maxblur strategy
        ddblur: True or False: if True and blur=True, use the blur strategy proposed in DDNeRF
    Return: 
        samples: [batchsize, N_samples]: sampling results on each ray in fine stage
    '''
    if blur:
        weights_pad = torch.cat([
            weights[..., :1],
            weights,
            weights[..., -1:],
        ],
            axis=-1)

        weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
        weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

        if not ddblur:
            weights = weights_blur + 0.001

        else:
            prev = weights_pad[..., :-2]
            next = weights_pad[..., 2:]

            weights = 0.8*weights + 0.1*prev + 0.1*next + 0.01

    position = torch.argmax(weights, dim = -1)
    below = torch.max(torch.zeros_like(position), position-1)
    above = torch.min((weights.shape[-1]-1) * torch.ones_like(position), position+1)
    inds_g = torch.stack([below, above], -1)

    bins_g0 = torch.gather(bins, 1, inds_g)
    bins_g = torch.zeros_like(bins_g0)

    p = 0.75 # p is to adjust how the sampling points distributed around max w_i. 
    bins_g[..., 0] = p * bins_g0[..., 0] + (1-p) * bins_g0[..., 1]
    bins_g[..., 1] = (1-p) * bins_g0[..., 0] + p * bins_g0[..., 1]
    
    # We choose to sample as below:
    u1 = (torch.linspace(0, 1, int(N_samples / 2), device= weights.device))**0.5
    u2 = (torch.linspace(1, 0, int(N_samples / 2), device= weights.device))**0.5
    u = torch.cat([0.5 * u1, 1 - 0.5 * u2], dim = -1)

    samples = bins_g[..., 0].unsqueeze(-1) + u * (bins_g[..., 1] - bins_g[..., 0]).unsqueeze(-1)

    return samples

def normal_sample_pdf(bins, weights, N_samples, det=True, blur = False, ddblur=False):
    '''
    Use {wi} to find a normal distribution to guide sampling. Results are always poor, just designed for comparison.
    Args:
        bins: [batchsize, n_sample_coarse]: the coarse {t_i} sampled uniformly on each ray
        weights: [batchsize, n_sample_coarse]: the coarse {w_i} on each ray
        N_samples: a number: the number of sampling points in fine stage
        det: True or False: choose different strategy when doing inverse transform sampling
        blur: True or False: if True, use the Maxblur strategy
        ddblur: True or False: if True and blur=True, use the blur strategy proposed in DDNeRF
    Return: 
        samples: [batchsize, N_samples]: sampling results on each ray in fine stage
    '''
    if blur:
        weights_pad = torch.cat([
            weights[..., :1],
            weights,
            weights[..., -1:],
        ],
            axis=-1)
        weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
        weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

        if ddblur:
            weights = weights_blur + 0.001

        else:
            prev = weights_pad[..., :-2]
            next = weights_pad[..., 2:]

            weights = 0.8*weights + 0.1*prev + 0.1*next + 0.01
    
    if det:
        u = torch.linspace(
            0.0, 1.0, steps=N_samples, dtype=weights.dtype, device=weights.device
        )
        u = u.expand(list(weights.shape[:-1]) + [N_samples])
    else:
        s = 1 / N_samples
        u = (torch.arange(N_samples, device=weights.device) * s).expand(list(weights.shape[:-1]) + [N_samples])
        u = u + (torch.rand(weights.shape[0], N_samples, device=weights.device)/((1/s) + 1e-5))
        u = torch.minimum(u, torch.tensor(0.9999))
    


    # Find the mean and variance of {w_i}
    mean = torch.sum(bins * weights, -1, keepdim=True) / (torch.sum(weights, -1, keepdim=True) + 1e-6)
    vars = torch.sum((bins - mean)**2 * weights, -1, keepdim=True) / (torch.sum(weights, -1, keepdim=True)+1e-6)
    std = torch.sqrt(vars)
    
    # inverse transform sampling
    ori_x = 0.5 * (1+torch.erf(u/torch.sqrt(torch.tensor(2., device=u.device))))
    samples = std * ori_x + mean
    samples = torch.clip(samples, 2., 6.)

    return samples