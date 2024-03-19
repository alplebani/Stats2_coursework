#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
import argparse
import time
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Helpers.HelperFunctions import MH, prior

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
    # parser.add_argument('-c', '--cov', help='Covariance', required=False, default=[[1.5,0.],[0.,1.5]],)
    
    if not os.path.exists('Plots'):
        os.makedirs('Plots')
    
    
    args = parser.parse_args()
    
    n_samples = args.nsamples
    burn_in = args.burnin
    range_alpha = args.a_range
    range_beta = args.b_range
    starting = args.starting
    
    # check that alpha, beta are in increasing order
    # for i in [range_alpha, range_beta]:
    
    data = np.loadtxt("lighthouse_flash_data.txt")
    
    positions = data[:, 0]
    
    covQ = [[1.5,0.],[0.,1.5]]
    
    priors = [lambda x: prior(x, range_alpha[0], range_beta[1]),
              lambda x:prior(x, range_beta[0], range_beta[1])]
    
    MC_chain, accepted = MH(data=positions, priors=priors, burn_in=burn_in, cov_Q=covQ, n=n_samples, starting=starting)
    
    print(f'Accepted = {accepted}')
    
    alphas = MC_chain[:, 0]
    betas = MC_chain[:, 1]
    
    alpha_mean = np.mean(alphas)
    alpha_std = np.std(alphas)
    beta_mean = np.mean(betas)
    beta_std = np.std(betas)
    
    print(f'Alpha = {alpha_mean} +- {alpha_std}')
    print(f'Beta = {beta_mean} +- {beta_std}')

    
    plt.figure()
    plt.hist2d(alphas, betas, bins=20, density=True)
    plt.xlabel("alpha")
    plt.ylabel("beta")
    plt.title("Joint Posterior Distribution")
    plt.savefig('Plots/Joint_posterior.pdf')
    print("=======================================")
    print('Saving plot at Plots/Joint_posterior.pdf')
    

    # Marginal histograms
    plt.figure()
    plt.hist(alphas, bins=30)
    plt.xlabel("alpha")
    plt.ylabel("Probability")
    plt.title("Marginal Posterior of alpha")
    plt.savefig('Plots/alpha_posterior.pdf')
    print("=======================================")
    print('Saving plot at Plots/alpha_posterior.pdf')
    
    plt.figure()
    plt.hist(betas, bins=30)
    plt.xlabel("beta")
    plt.ylabel("Probability")
    plt.title("Marginal Posterior of beta")
    plt.savefig('Plots/beta_posterior.pdf')
    print("=======================================")
    print('Saving plot at Plots/beta_posterior.pdf')
    
    fig, ax = plt.subplots()

    first_n_steps = 10000

    ax.plot(MC_chain[0:first_n_steps,0], label=r'$\alpha$')
    ax.plot(MC_chain[0:first_n_steps,1], label=r'$\beta$')
    plt.axhline(y = alpha_mean, color='r', linestyle='--', label=r'$\langle \alpha \rangle$')
    plt.axhline(y = beta_mean, color='black', linestyle='--', label=r'$\langle \beta \rangle$')
    plt.title('Trace plot')
    plt.legend(loc='upper left')
    plt.savefig('Plots/trace_plot.pdf')
    print("=======================================")
    print('Saving plot at Plots/trace_plot.pdf')
  
        
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