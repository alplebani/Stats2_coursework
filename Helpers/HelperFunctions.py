import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def Likelihood(x, params):
    '''
    Likelihood for flash location
    '''
    
    alpha = params[0]
    beta = params[1]
    
    return beta/(np.pi*(beta**2 + (x - alpha)**2))

    
def prior(x, lower, upper):
    '''
    Uniform prior for alpha and beta
    '''
    
    return 1.0/(upper - lower) if lower <= x <= upper else 0.0



def MH(data, priors, n, burn_in, cov_Q, starting):
    '''
    Metropolis-Hasting algorithm for sampling
    '''
    
    accepted = 0
    MC_chain = np.zeros((n, 2))
    MC_chain[0] = (starting)        
    
    for i in range(1, n):
        x_current = MC_chain[i-1]
        
        x_proposed = multivariate_normal.rvs(x_current, cov_Q)
             
        while x_proposed[1] <= 0:
            x_proposed = multivariate_normal.rvs(x_current, cov_Q)
            
        likelihood_proposed = Likelihood(data, x_proposed)
        likelihood_current = Likelihood(data, x_current)
        log_liks = np.log(likelihood_proposed) - np.log(likelihood_current) 
        log_priors = np.log(priors[0](x_proposed[0])) - np.log(priors[0](x_current[0])) + np.log(priors[1](x_proposed[1])) - np.log(priors[1](x_current[1]))
        
        log_a = log_liks + log_priors
        
        # breakpoint()
        
        if np.log(np.random.uniform()) < np.sum(log_a):
            # breakpoint()
            x_new = x_proposed
            accepted += 1
        else: 
            x_new = x_current
            
        if i>= burn_in:
        
            MC_chain[i] = x_new
            
    return MC_chain, accepted


def Likelihood_model2(x, params):
    '''
    Likelihood for part vi), using also intensity
    '''
    
    if len(params) != 3:
        raise ValueError('Error! Required 3 parameters for the Likelihood: alpha, beta, I0')
    locations = x[:, 0]
    intensities = x[:, 1]
    alpha = params[0]
    beta = params[1]
    I0 = params[2]
    
    loc_likelihood = Likelihood(x, params)
    int_likelihood = np.exp(-(np.log(intensities) - np.log(I0) + np.log(1.0) / (beta**2 + (locations - alpha)**2))**2 / (2.0)) / np.sqrt(2.0 * np.pi) # sigma set to 1
    
    return loc_likelihood*int_likelihood


