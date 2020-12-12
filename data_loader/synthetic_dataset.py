import logging
import numpy as np
import os,sys
sys.path.append(os.getcwd())
from helpers.torch_utils import set_seed
from helpers.analyze_utils import plot_timeseries


class SyntheticDataset(object):
    """
    A Class for generating data.
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, num_X, num_Z,  num_samples, max_lag, noise_distribution='normal'):
        """
        Args:
            num_X: number of observed variables, i.e. m in the formulation
            num_Z: number of latent variables
            max_lag: the maximal lag
            
            noise_distribution: the distribution of noise, default np.random.normal()
            
        """
        
        self.num_X = num_X
        self.num_Z = num_Z
        self.num_samples = num_samples
        self.max_lag = max_lag

        self.noise_distribution = noise_distribution

        self._setup()
        self._logger.debug('Finished setting up dataset class')

    def _setup(self):
        self.A, self.W, self.groudtruth = SyntheticDataset.simulate_random_dag(self.num_X, self.num_Z, self.max_lag)

        self.X, self.Z = SyntheticDataset.simulate_sem(self.W, self.num_X, self.num_Z, self.max_lag, self.num_samples, self.noise_distribution)


    @staticmethod
    def simulate_random_dag(num_X, num_Z, max_lag):
        """ Simulate random DAG.
            For each lag, we consider the following step:
                We generate a binary matrix A with shape (num_X+num_Z, num_X+num_Z) to store the connection of variables.
                A[i,j]=1 means j->i.

                Z causes X but X don't causes Z, Zi and Zj are independent for any i,j.
                So A_xx is a lower triangular matrix, A_xz is a random matrix with shape (num_X,num_Z), A_zx is a zero matrix and A_zz is a diagonal matrix.


            Args:
                num_X: number of observed variables
                num_Z: number of latent variables
                max_lag: number of maximal lag

            Returns:
                A: a list with size max_lag, each element is a binary adjacency matrix  of X and Z              
                W: a list with size max_lag, each element is a weighted matrix of X and Z
                groudtruth: a list with size max_lag, each element is a binary adjacency matrix of X


        """
        A,W = [],[]
        groudtruth = []
        for _ in range(max_lag):
            A_xx = np.zeros(shape=(num_X,num_X)) # x->x
            while abs(A_xx).sum() < num_X: #  used for controlling the sparsity of generated DAG,  (2*num_X)/(num_x(num_x-1)/2)
                A_xx = np.tril( np.random.randint(low=0,high=2, size=(num_X,num_X)), k=0) # set k=-1 to exclude the diagonal elements, but in our case time-series is usually self-caused, so we keep the diagonal elements.

            groudtruth.append(A_xx)

            A_xz = np.random.randint(low=0,high=2,size=(num_X,num_Z)) # z->x
            A_zx = np.zeros(shape=(num_Z,num_X)) # x->z
            A_zz = np.diag( np.random.randint(low=0, high=2, size=num_Z) ) # z->z

            tmp_A = np.concatenate( (
                np.concatenate((A_xx,A_xz),axis=1), 
                np.concatenate((A_zx,A_zz),axis=1) ), axis=0) 
            A.append(tmp_A)
            
            tmp_W = np.random.randn(num_X+num_Z,num_X+num_Z) * tmp_A
            # tmp_W = np.random.normal(loc = np.zeros((num_X+num_Z,num_X+num_Z)), scale = np.ones((num_X+num_Z,num_X+num_Z))) * tmp_A

            W.append(tmp_W)

        return A, W, groudtruth


    @staticmethod         
    def simulate_sem(W, num_X, num_Z, max_lag, num_samples, noise_distribution):

        # Make data stable
        burn_in = 5000
        T = num_samples + burn_in
        
        N = num_X+num_Z

        if noise_distribution == 'normal':
            # Initialize Gaussian noise for observed varaibles and hidden variables
            data = np.random.normal(size=(T,N)) 
        else:
                raise ValueError('Undefined noise_distribution type')

        # Consider a time series coming from the following data generating process
        for t in range(max_lag+1, T+1):

            tmp_X = np.zeros(num_X)
            tmp_Z = np.zeros(num_Z)
            
            for lag in range(1,max_lag+1):
                

                coefficient = W[lag-1]

                coefficient_11 = coefficient[0:num_X,0:num_X] # i.e., A_11⊙W_11
                coefficient_12 = coefficient[0:num_X,num_X:N] # i.e., A_12⊙W_12
                coefficient_22 = coefficient[num_X:N,num_X:N] # i.e., A_22⊙W_22

                
                tmp_X +=  (  np.matmul(data[t-lag-1,0:num_X],coefficient_11.T) +
                             np.matmul(data[t-lag-1,num_X:N],coefficient_12.T) )
                            
                if num_Z:
                    tmp_Z +=   np.matmul(data[t-lag-1,num_X:N],coefficient_22.T) 

            # we normalize Z AND X, in case it become larger and larger when coefficient>1.

            tmp_X = tmp_X/(np.max(abs(tmp_X))  + 1e-31) # to avoid 0/0
            
            # data[t,0:num_X] = tmp_X + np.random.normal(size=num_X)
            data[t-1,0:num_X] += tmp_X 
            if num_Z:
                tmp_Z = tmp_Z/(np.max(abs(tmp_Z)) + 1e-31 ) # to avoid 0/0
                # data[t,num_X:N] = tmp_Z + np.random.normal(size=num_Z)
                data[t-1,num_X:N] += tmp_Z 

        X = data[-num_samples:,0:num_X]
        Z = data[-num_samples:,num_X:N]

        
        return  X, Z

    def save_dataset(self, output_dir):
        
        # Save the A,W,X,Z
        np.save(output_dir+'/groudtruth_A.npy',self.A)
        np.save(output_dir+'/groudtruth_W.npy',self.W)
        np.savetxt(output_dir+'/data_X.csv',self.X,delimiter=',')
        np.savetxt(output_dir+'/data_Z.csv',self.Z,delimiter=',') 



if __name__ == '__main__':

    set_seed(2020)
    
    num_X, num_Z,  num_samples, max_lag, noise_distribution = 2, 2, 100, 1 ,'normal'

    dataset = SyntheticDataset(num_X, num_Z,  num_samples, max_lag, noise_distribution)

    print(dataset.X.shape)
    print(dataset.W)
    print(dataset.groudtruth)
    plot_timeseries(dataset.X[0:100],'X')
    if num_Z:
        plot_timeseries(dataset.Z[0:100],'Z') 


