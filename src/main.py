#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
import argparse
import time
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Helpers.HelperFunctions import MH, prior, prior_intensity
from Helpers.HelperFunctions import plot_histo_2d, plot_marginal, plot_trace
# Set matplotlib style for plots
plt.style.use('mphil.mplstyle')


def main():
    
    # Command-line options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--plots', help='Flag: if selected, will show the plots instead of only saving them', required=False, action='store_true')
    parser.add_argument('-n', '--nsamples', help='Number of MC samples you want to generate', required=False, default=10000, type=int)
    parser.add_argument('-b', '--burnin', help='Burn in: number of first samples that are not considered for the chain', required=False, default=100, type=int)
    parser.add_argument('--a_range', help='Range of alpha', required=False, default=[-5,5], nargs='+', type=float)
    parser.add_argument('--b_range', help='Range of beta', required=False, default=[0,5], nargs='+', type=float)
    parser.add_argument('-s', '--starting', help='Starting values for the chain', default=[0.2, 0.2], nargs='+', required=False, type=float)
    parser.add_argument('-i', '--intensity', help='Mean of log-I distribution for I0', required=False, default=1., type=float)
    
    # parser.add_argument('-c', '--cov', help='Covariance', required=False, default=[[1.5,0.],[0.,1.5]],)
    
    if not os.path.exists('Plots'):
        os.makedirs('Plots')
    
    args = parser.parse_args()
    
    n_samples = args.nsamples
    burn_in = args.burnin
    range_alpha = args.a_range
    range_beta = args.b_range
    starting = args.starting
    
    start_loc = [starting[0], starting[1]]
    
    # check that alpha, beta are in increasing order
    # for i in [range_alpha, range_beta]:
    
    data = np.loadtxt("lighthouse_flash_data.txt")
    
    print("=======================================")
    print('Executing part V)')
    print("---------------------------------------")
    
    covQ = [[1.5,0.],[0.,1.5]]
    
    priors = [lambda x: prior(x, range_alpha[0], range_beta[1]),
              lambda x:prior(x, range_beta[0], range_beta[1])]
    
    MC_chain, accepted = MH(data=data, priors=priors, burn_in=burn_in, cov_Q=covQ, n=n_samples, starting=start_loc)
    
    print(f'Accepted = {accepted}')
    
    alphas = MC_chain[:, 0]
    betas = MC_chain[:, 1]
    
    alpha_mean = np.mean(alphas)
    alpha_std = np.std(alphas)
    beta_mean = np.mean(betas)
    beta_std = np.std(betas)
    
    print(f'Alpha = {alpha_mean:.4g} +- {alpha_std:.4g}')
    print(f'Beta = {beta_mean:.4g} +- {beta_std:.4g}')

    # Joint histogram
    plot_histo_2d(alphas, betas, 'alpha', 'beta', 'position')    

    # Marginal histograms
    plot_marginal(alphas, 'alpha', custom_name='position')
    plot_marginal(betas, 'beta', custom_name='position')

    means = [alpha_mean, beta_mean]

    plot_trace(MC_chain, means, ['alpha', 'beta'], size=2, colors=['r', 'blue'], custom_name='position', latexs=[r'$\langle \alpha \rangle$', r'$\langle \beta \rangle$'])
    
    print("=======================================")
    print('Executing part VII)')
    print("---------------------------------------")
    
    mu_int = args.intensity
  
    priors.append(lambda x: prior_intensity(x, mu=np.log(mu_int)))
    
    covQ = [[1.5,0,0],[0,1.5,0],[0,0,1.5]]
    
    MC_chain_2, accepted_int = MH(data=data, priors=priors, burn_in=burn_in, cov_Q=covQ, n=n_samples, starting=starting)
    
    print(f'Accepted = {accepted_int}')
    
    alphas_int = MC_chain_2[:, 0]
    betas_int = MC_chain_2[:, 1]
    intensities = MC_chain_2[:, 2]
    
    alpha_int_mean = np.mean(alphas_int)
    alpha_int_std = np.std(alphas_int)
    beta_int_mean = np.mean(betas_int)
    beta_int_std = np.std(betas_int)
    int_mean = np.mean(intensities)
    int_std = np.std(intensities)
    
    print(f'Alpha = {alpha_int_mean:.4g} +- {alpha_int_std:.4g}')
    print(f'Beta = {beta_int_mean:.4g} +- {beta_int_std:.4g}')
    print(f'I0 = {int_mean:.4g} +- {int_std:.4g}')
        
    plot_histo_2d(alphas_int, betas_int, 'alpha', 'beta', 'part7')   
    plot_histo_2d(alphas_int, intensities, 'alpha', 'I0', 'part7')
    plot_histo_2d(betas_int, intensities, 'beta', 'I0', 'part7')
    
    plot_marginal(alphas_int, 'alpha', 'part7')
    plot_marginal(betas_int, 'beta', 'part7')
    plot_marginal(intensities, 'I0', 'part7') 
    
    means = [alpha_int_mean, beta_int_mean, int_mean]
    
    plot_trace(MC_chain_2, means, ['alpha', 'beta', 'I0'], size=3, colors=['r', 'blue', 'black'], custom_name='part7', latexs=[r'$\langle \alpha \rangle$', r'$\langle \beta \rangle$', r'$\langle I0 \rangle$'])
    
        
    if args.plots: # display plots only if the --plots is used
        plt.show()


if __name__ == "__main__":
    print("=======================================")
    print("Initialising coursework")
    print("=======================================")
    start_time = time.time()
    main()
    end_time = time.time()
    print("=======================================")
    print("Coursework finished. Exiting!")
    print("Time it took to run the code : {} seconds". format(end_time - start_time))
    print("=======================================")