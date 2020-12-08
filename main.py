#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pytz import timezone
from datetime import datetime
import numpy as np
import torch


from data_loader.synthetic_dataset import SyntheticDataset
from data_loader.real_dataset import RealDataset
from models.TimeLatent import TimeLatent
from trainers.trainer import Trainer


from helpers.config_utils import save_yaml_config, get_args
from helpers.log_helper import LogHelper
from helpers.torch_utils import set_seed
from helpers.dir_utils import create_dir
from helpers.analyze_utils import  plot_timeseries, plot_losses, plot_recovered_graph, plot_ROC_curve, AUC_score, F1


def synthetic():
    
    np.set_printoptions(precision=3)
    
    # Get arguments parsed
    args = get_args()

    # Setup for logging
    output_dir = 'output/{}'.format(datetime.now(timezone('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
    create_dir(output_dir)
    LogHelper.setup(log_path='{}/training.log'.format(output_dir),
                    level_str='INFO')
    _logger = logging.getLogger(__name__)
    
    # Save the configuration for logging purpose
    save_yaml_config(args, path='{}/config.yaml'.format(output_dir))


    # Reproducibility
    set_seed(args.seed)

    # Get dataset
    dataset = SyntheticDataset(args.num_X, args.num_Z,  args.num_samples, args.max_lag)
    # Save dataset
    dataset.save_dataset( output_dir=output_dir)
    _logger.info('Finished generating dataset')

    # Look at data
    _logger.info('The shape of observed data: {}'.format(dataset.X.shape))
    plot_timeseries(dataset.X[-150:],'X',display_mode=False,save_name=output_dir+'/timeseries_X.png')
    plot_timeseries(dataset.Z[-150:],'Z',display_mode=False,save_name=output_dir+'/timeseries_Z.png')

    # Init model
    model = TimeLatent(args.num_X, args.max_lag, args.num_samples, args.device, args.prior_rho_A, args.prior_sigma_W, args.temperature, args.sigma_Z, args.sigma_X)

    trainer = Trainer(args.learning_rate, args.num_iterations, args.num_output)

    trainer.train_model(model=model, X = torch.tensor(dataset.X,dtype=torch.float32,device=args.device), output_dir=output_dir)

    plot_losses(trainer.train_losses,display_mode=False,save_name=output_dir+'/loss.png')

    # Save result
    trainer.log_and_save_intermediate_outputs()

    _logger.info('Finished training model')

    # Calculate performance
    
    estimate_A = model.posterior_A.probs[:,:args.num_X,:args.num_X].cpu().data.numpy() # model.posterior_A.probs is shape with (max_lag,num_X+num_Z,num_X+num_Z)
    groudtruth_A = np.array(dataset.groudtruth) # groudtruth is shape with (max_lag,num_X,num_X)

    Score = AUC_score(estimate_A.T,groudtruth_A.T)
    _logger.info('\n        fpr:{} \n        tpr:{}\n thresholds:{}\n AUC:{}'.format(Score['fpr'],Score['tpr'],Score['thresholds'],Score['AUC']))

    plot_ROC_curve(estimate_A.T,groudtruth_A.T,display_mode=False,save_name=output_dir+'/ROC_Curve.png')
    
    for t in range(0,11):
        _logger.info('Under threshold:{}'.format(t/10))
        _logger.info(F1(estimate_A.T,groudtruth_A.T,threshold=t/10))

    # Visualizations
    for k in range(args.max_lag):
        # Note that in our implementation, A_ij=1 means j->i, but in the plot_recovered_graph A_ij=1 means i->j, so transpose A
        plot_recovered_graph(estimate_A[k].T,groudtruth_A[k].T,title='Lag = {}'.format(k+1),display_mode=False,save_name=output_dir+'/lag_{}.png'.format(k))


    _logger.info('All Finished!')



def real():
    
    np.set_printoptions(precision=3)
    
    # Get arguments parsed
    args = get_args()

    # Setup for logging
    output_dir = 'output/real_{}'.format(datetime.now(timezone('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
    create_dir(output_dir)
    LogHelper.setup(log_path='{}/training.log'.format(output_dir),
                    level_str='INFO')
    _logger = logging.getLogger(__name__)
    
    # Save the configuration for logging purpose
    save_yaml_config(args, path='{}/config.yaml'.format(output_dir))


    # Reproducibility
    set_seed(args.seed)

    # Get dataset
    dataset = RealDataset()

    # Look at data
    _logger.info('The shape of observed data: {}'.format(dataset.stock.shape))
    plot_timeseries(dataset.stock[-150:],'stock',display_mode=False,save_name=output_dir+'/timeseries_stock.png')

    # Set parameters
    num_samples, num_X = dataset.stock.shape
    temperature = 2.0
    max_lag = 1
    prior_rho_A = 0.5
    prior_sigma_W = 1.0
    sigma_Z = 1.0
    sigma_X = 1.0
    num_iterations = 2000

    # Log the parameters
    _logger.info("num_X:{},max_lag:{},num_samples:{},args.device:{},prior_rho_A:{},prior_sigma_W:{},temperature:{},sigma_Z:{},sigma_X:{},num_iterations:{}".format(num_X,max_lag,num_samples,args.device,prior_rho_A,prior_sigma_W,temperature,sigma_Z,sigma_X,num_iterations))
    # Init model
    model = TimeLatent(num_X=num_X,max_lag=max_lag,num_samples=num_samples,device=args.device,prior_rho_A=prior_rho_A,prior_sigma_W=prior_sigma_W,temperature=temperature,sigma_Z=sigma_Z,sigma_X=sigma_X)
    trainer = Trainer(learning_rate=args.learning_rate, num_iterations=num_iterations,num_output=args.num_output)

    trainer.train_model(model=model, X = torch.tensor(dataset.stock,dtype=torch.float32,device=args.device), output_dir=output_dir)

    plot_losses(trainer.train_losses,display_mode=False,save_name=output_dir+'/loss.png')

    # Save result
    trainer.log_and_save_intermediate_outputs()

    _logger.info('Finished training model')

    
    estimate_A = model.posterior_A.probs.cpu().data.numpy()

    # Visualizations
    for k in range(max_lag):
        # Note that in our implementation, A_ij=1 means j->i, but in the plot_recovered_graph A_ij=1 means i->j, so transpose A
        plot_recovered_graph(estimate_A[k].T,W=None,title='Lag = {}'.format(k+1),display_mode=False,save_name=output_dir+'/lag_{}.png'.format(k))

    _logger.info('All Finished!')


if __name__ == '__main__':

    synthetic()

    # real()
