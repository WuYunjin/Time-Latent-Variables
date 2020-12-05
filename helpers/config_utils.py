import sys
import yaml
import argparse
from helpers.torch_utils import get_device


def load_yaml_config(path, skip_lines=0):
    with open(path, 'r') as infile:
        for i in range(skip_lines):
            # Skip some lines (e.g., namespace at the first line)
            _ = infile.readline()

        return yaml.safe_load(infile)


def save_yaml_config(config, path):
    with open(path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def get_args():
    parser = argparse.ArgumentParser()

    ##### General settings #####
    parser.add_argument('--seed',
                        type=int,
                        default=2020,
                        help='Random seed')

    parser.add_argument('--device',
                        default=get_device(),
                        help='Running device')

    ##### Dataset settings #####
    parser.add_argument('--num_X',
                        type=int,
                        default=5,
                        help='Number of observed variables')

    parser.add_argument('--num_Z',
                        type=int,
                        default=5,
                        help='Number of latent variables')

    parser.add_argument('--num_samples',
                        type=int,
                        default=500,
                        help='Number of sample size') # different sample size may need a different learning rate to train.
    
    parser.add_argument('--max_lag',
                        type=int,
                        default=1,
                        help='Number of maximal time lag')



    ##### Model settings #####
    parser.add_argument('--prior_rho_A',
                        type=float,
                        default=0.6,
                        help='the parameter of Bernoulli distribution, which is the prior over A_k,ij')

    parser.add_argument('--prior_sigma_W',
                        type=float,
                        default=1.0,
                        help='the standard deviation parameter of Normal distribution, which is the prior over W_k,ij')

    parser.add_argument('--temperature',
                        type=float,
                        default=2.0,
                        help='the temperature parameter of Gumbel-Bernoulli distribution')
    
    parser.add_argument('--sigma_Z',
                        type=float,
                        default=1.0,
                        help='the standard deviation parameter of Normal distribution over latent variables Z')

    parser.add_argument('--sigma_X',
                        type=float,
                        default=0.1,
                        help='the standard deviation parameter of Normal distribution over observed variables X')

    # parser.add_argument('--threshold_A',
    #                     type=float,
    #                     default=0.1,
    #                     help='the threshold parameter of determinating whether there is a edge from i to j')

                
    
    ##### Training settings #####
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-3,
                        help='Learning rate for optimizer') # sensitive


    parser.add_argument('--num_iterations',
                        type=int,
                        default=3500,
                        help='Number of iterations')

    parser.add_argument('--num_output',
                        type=int,
                        default=10,
                        help='Number of iterations to display information')


    ##### Other settings #####

    return parser.parse_args(args=sys.argv[1:])
