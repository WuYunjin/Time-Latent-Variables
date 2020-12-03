import logging
import numpy as np
import torch
from itertools import chain
from torch import optim
from torch.nn.utils import clip_grad_value_


class Trainer(object):
    """
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, learning_rate, num_iterations, num_output):

        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.num_output = num_output

    def train_model(self, model, X,  output_dir):
        self.output_dir = output_dir
        m = X.shape[1]
        train_losses = []
        optimizer = optim.Adam(params=[model.posterior_A.probs, model.posterior_W.loc ,model.posterior_W.scale],lr=self.learning_rate)
        for iteration in range(self.num_iterations):

            
            optimizer.zero_grad()

            loss = model.loss(X)
            loss.backward(retain_graph=True)

            # Clipping Gradient for parameters
            clip_grad_value_([model.posterior_A.probs, model.posterior_W.loc ,model.posterior_W.scale],clip_value=2.0)

            optimizer.step()
            train_losses.append(loss.item())
            if(iteration% self.num_output==0):
                self._logger.info("Iteration {} , loss:{}".format(iteration,loss.item()))

            with torch.no_grad():
                # model.posterior_A.probs should  >=0, but after the gradient decent  it may lead to <0 and therefore we clamp it.
                model.posterior_A.probs.data = torch.clamp(model.posterior_A.probs,min=0.0)

                # replace the nan in W, maybe unnecessary
                for i in range(m):
                    tmp = model.posterior_W.scale[:,m+i,m+i].clone().detach()
                    model.posterior_W.scale[:,m+i,:] = 0.0
                    model.posterior_W.scale[:,m+i,m+i] = tmp

                    tmp = model.posterior_W.loc[:,m+i,m+i].clone().detach()
                    model.posterior_W.loc[:,m+i,:] = 0.0
                    model.posterior_W.loc[:,m+i,m+i] = tmp
                # print('test')
                
        self.train_losses = train_losses
        self.estimate_A = model.posterior_A.probs.cpu().data.numpy()
        self.estimate_W_loc = model.posterior_W.loc.cpu().data.numpy()
        self.estimate_W_scale = model.posterior_W.scale.cpu().data.numpy()

    def log_and_save_intermediate_outputs(self):
        # may want to save the intermediate results
        
        # Save the loss
        np.savetxt(self.output_dir+'/loss.txt',self.train_losses) # np.savetxt Only works for 1D or 2D array,np.save works for higher dimension array.

        # Save the model.posterior_A.probs, model.posterior_W.loc ,model.posterior_W.scale
        np.save(self.output_dir+'/estimate_A.npy',self.estimate_A) # np.load(file=self.output_dir+'/estimate_A.npy')
        np.save(self.output_dir+'/estimate_W_loc.npy',self.estimate_W_loc)
        np.save(self.output_dir+'/estimate_W_scale.npy',self.estimate_W_scale)