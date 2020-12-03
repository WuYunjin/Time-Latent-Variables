#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pytz import timezone
from datetime import datetime
import numpy as np
import torch


from data_loader.synthetic_dataset import SyntheticDataset
from models.TimeLatent import TimeLatent
from trainers.trainer import Trainer


from helpers.config_utils import save_yaml_config, get_args
from helpers.log_helper import LogHelper
from helpers.torch_utils import set_seed
from helpers.dir_utils import create_dir
from helpers.analyze_utils import  plot_timeseries, plot_losses, plot_recovered_graph, plot_ROC_curve, AUC_score


def main():
    # Get arguments parsed
    args = get_args()

    # Setup for logging
    output_dir = 'output/{}'.format(datetime.now(timezone('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
    create_dir(output_dir)
    LogHelper.setup(log_path='{}/training.log'.format(output_dir),
                    level_str='INFO')
    _logger = logging.getLogger(__name__)

    # Reproducibility
    set_seed(args.seed)

    # Get dataset
    dataset = SyntheticDataset(args.num_X, args.num_Z,  args.num_samples, args.max_lag)
    # Save dataset
    dataset.save_dataset( output_dir=output_dir)
    _logger.info('Finished generating dataset')

    # Look at data
    _logger.info('The shape of observed data: {}'.format(dataset.X.shape))
    plot_timeseries(dataset.X[0:100],'X',display_mode=False,save_name=output_dir+'/timeseries_X.png')
    plot_timeseries(dataset.Z[0:100],'Z',display_mode=False,save_name=output_dir+'/timeseries_Z.png')

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

    AUC = AUC_score(estimate_A.T,groudtruth_A.T)
    _logger.info('AUC Score :{}'.format(AUC))

    plot_ROC_curve(estimate_A.T,groudtruth_A.T,display_mode=True,save_name=output_dir+'/ROC_Curve.png')
    
    
    # Visualizations
    # estimate_A= (abs(estimate_A)> args.threshold_A).astype(int)
    for k in range(args.max_lag):
        # Note that in our implementation, A_ij=1 means j->i, but in the plot_recovered_graph A_ij=1 means i->j, so transpose A
        plot_recovered_graph(estimate_A[k].T,groudtruth_A[k].T,title='Lag = {}'.format(k+1),display_mode=True,save_name=output_dir+'/lag_{}.png'.format(k))


    # Save the configuration for logging purpose
    save_yaml_config(args, path='{}/config.yaml'.format(output_dir))

    _logger.info('All Finished!')



if __name__ == '__main__':

    main()
