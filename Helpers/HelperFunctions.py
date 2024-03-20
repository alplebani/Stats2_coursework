import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def Likelihood(data, params):
    '''
    Likelihood for flash location
    
    Args: data (numpy.ndarray): Array containing flash location data
    params (list): Lisy containing alpha and beta parameters
    
    Returns:
    float: Likelihood value
    '''
    
    x = data[:, 0]
    alpha = params[0]
    beta = params[1]
    
    return beta/(np.pi*(beta**2 + (x - alpha)**2))

    
def prior(x, lower, upper):
    '''
    Uniform prior for alpha and beta.
    
    Args:
    x (float): Value at which to evaluate the prior.
    lower (float): Lower bound of the prior.
    upper (float): Upper bound of the prior.
    
    Returns:
    float: Prior probability density.
    '''
    
    return 1.0/(upper - lower) if lower <= x <= upper else 0.0


def Likelihood_model2(x, params):
    '''
    Likelihood for part vi), using also intensity
    
    Args:
    x (numpy.ndarray): Array containing flash location and intensity data.
    params (list): List containing alpha, beta, and I0 parameters.
    
    Returns:
    float: Likelihood value.
    
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


def prior_intensity(I, mu, sigma=1):
    '''
    Prior for I0: log-normal prior.
    
    Args:
    I (float): Intensity value.
    mu (float): Mean of the log-normal distribution.
    sigma (float): Standard deviation of the log-normal distribution.
    
    Returns:
    float: Prior probability density.
    
    '''
    
    if I > 0:
        return 1.0/(I * np.sqrt(2 * np.pi) * sigma) * np.exp(-(np.log(I) - mu)**2 / (2 * sigma**2))
    else: 
        return 0.

def MH(data, priors, n, burn_in, cov_Q, starting):
    '''
    Metropolis-Hasting algorithm for sampling
    
    Args:
    data (numpy.ndarray): Array containing flash location data.
    priors (list): List of prior functions for each parameter.
    n (int): Number of samples to generate.
    burn_in (int): Number of initial samples not considered for the chain.
    cov_Q (list): Covariance matrix for proposing new samples.
    starting (list): Starting values for the chain.
    
    Returns:
    numpy.ndarray: Markov chain samples.
    int: Number of accepted samples.
    '''
    
    if len(priors) != len(starting):
        raise ValueError('Error! Need priors to be of the same length of the starting values')
    
    size = len(priors)
    
    accepted = 0
    MC_chain = np.zeros((n, len(priors)))
    MC_chain[0] = (starting)    
    log_priors = 0    
    
    for i in range(1, n):
        x_current = MC_chain[i-1]
        
        x_proposed = multivariate_normal.rvs(x_current, cov_Q)
            
        while x_proposed[1] <= 0:
            x_proposed = multivariate_normal.rvs(x_current, cov_Q)
            
        log_priors = np.log(priors[0](x_proposed[0])) - np.log(priors[0](x_current[0]))
        
        if size == 2: 
            likelihood_proposed = Likelihood(data, x_proposed)
            likelihood_current = Likelihood(data, x_current)
            
        elif size == 3:
            likelihood_proposed = Likelihood_model2(data, x_proposed)
            likelihood_current = Likelihood_model2(data, x_current)
            log_priors = log_priors + np.log(priors[2](x_proposed[2])) - np.log(priors[2](x_current[2]))
            # breakpoint()
        else:
            raise ValueError('Error! Only models with 2 or 3 parameters are accepted')
        
        log_liks = np.log(likelihood_proposed) - np.log(likelihood_current) 
        # breakpoint()
        
        log_a = log_liks + log_priors
        
        if np.log(np.random.uniform()) < np.sum(log_a):
            x_new = x_proposed
            accepted += 1
        else: 
            x_new = x_current
            
        if i>= burn_in:        
            MC_chain[i] = x_new

    return MC_chain, accepted


def plot_histo_2d(arr1, arr2, name1, name2, custom_name, bins=20):
    '''
    Function to plot the 2d histogram
    
    Args:
    arr1 (numpy.ndarray): Values for the x-axis.
    arr2 (numpy.ndarray): Values for the y-axis.
    name1 (str): Label for the x-axis.
    name2 (str): Label for the y-axis.
    custom_name (str): Custom name for svaing the plot.
    bins (int): Number of bins for the histogram.
    '''
    
    plt.figure(figsize=(15,10))
    plt.hist2d(arr1, arr2, bins=bins)
    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.title(f'Joint posterior for {name1} and {name2}')
    plt.savefig(f'Plots/Joint_{name1}_{name2}_{custom_name}.pdf')
    print("=======================================")
    print(f'Saving plot at Plots/Joint_{name1}_{name2}_{custom_name}.pdf')
    
def plot_marginal(arr, name, custom_name):
    '''
    Plot marginal pdf for one parameter
    
    Args:
    arr (numpy.ndarray): Array containing parameter values.
    name (str): Label for the parameter.
    custom_name (str): Custom name for saving the plot.
    '''
    
    plt.figure(figsize=(15,10))
    plt.hist(arr, bins=30)
    plt.xlabel(f'{name}')
    plt.ylabel(f'Probability')
    plt.title(f'Marginal Posterior of {name}')
    plt.savefig(f'Plots/{name}_posterior_{custom_name}.pdf')
    print("=======================================")
    print(f'Saving plot at Plots/{name}_posterior_{custom_name}.pdf')
    
def plot_trace(MC_chain, means, labels, size, colors, custom_name, latexs):
    '''
    Plot the trace for the MC chain
    
    Args:
        MC_chain (numpy.ndarray): The Markov Chain array containing sampled values.
        means (list): List of mean values for each parameter.
        labels (list): List of labels for parameters.
        size (int): Number of parameters.
        colors (list): List of colors for each parameter trace.
        custom_name (str): Custom name for saving the plot.
        latexs (list): List of latex labels for parameters.
    '''
    
    fig, ax = plt.subplots(figsize=(15,10))

    for i in range(size):
        ax.plot(MC_chain[:,i], label=labels[i])
        plt.axhline(y = means[i], color=colors[i], linestyle='--', label=latexs[i])
        
    plt.title('Trace plot')
    plt.legend(loc='upper right')
    plt.savefig(f'Plots/trace_plot_{custom_name}.pdf')
    print("=======================================")
    print(f'Saving plot at Plots/trace_plot_{custom_name}.pdf')