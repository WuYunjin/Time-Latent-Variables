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

    def __init__(self, num_X, num_Z,  num_samples, max_lag, noise_distribution='MoG', num_gaussian_component=1):
        """
        Args:
            num_X: number of observed variables, i.e. m in the formulation
            num_Z: number of latent variables
            max_lag: the maximal lag
            
            noise_distribution: the distribution of noise, default MoG with 1 component ,i.e. Gaussian noise
            
        """
        
        self.num_X = num_X
        self.num_Z = num_Z
        self.num_samples = num_samples
        self.max_lag = max_lag

        self.noise_distribution = noise_distribution
        self.num_gaussian_component = num_gaussian_component

        self._setup()
        self._logger.debug('Finished setting up dataset class')

    def _setup(self):
        self.A, self.W, self.groudtruth = SyntheticDataset.simulate_random_dag(self.num_X, self.num_Z, self.max_lag)

        self.X, self.Z = SyntheticDataset.simulate_sem(self.W, self.num_X, self.num_Z, self.max_lag, self.num_samples,noise_distribution=self.noise_distribution, num_gaussian_component=self.num_gaussian_component)


    @staticmethod
    def simulate_dag( num_variables, graph_type='ER', prob=0.3):
        # Referred from : https://github.com/xunzheng/notears/blob/master/notears/utils.py

        """Simulate random DAG with some expected number of edges.
        Args:
                num_variables (int): num of nodes
                prob (float): the probability of edges
                graph_type (str): ER, SF, BP
        Returns:
                B (np.ndarray): [d, d] binary adj matrix of DAG
        """

        import igraph as ig
        def _random_permutation(M):
            # np.random.permutation permutes first axis only
            P = np.random.permutation(np.eye(M.shape[0]))
            return P.T @ M @ P

        def _random_acyclic_orientation(B_und):
            return np.tril(_random_permutation(B_und), k=-1)

        def _graph_to_adjmat(G):
            return np.array(G.get_adjacency().data)

        if graph_type == 'ER':
            # Erdos-Renyi
            G_und = ig.Graph.Erdos_Renyi(n=num_variables, p=prob)
            B_und = _graph_to_adjmat(G_und)
            B = _random_acyclic_orientation(B_und)

        else:
            raise ValueError('unknown graph type')
        B_perm = _random_permutation(B)
        assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
        return B_perm

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
        for _ in range(max_lag): # Note that we only consider time-lagged effect, so max_lag should >= 1
            # A
            A_xx = np.zeros(shape=(num_X,num_X)) # x->x
            while abs(A_xx).sum() < 3: #  used for controlling the sparsity of generated DAG,at least 3 edges
                A_xx = SyntheticDataset.simulate_dag(num_X)
            groudtruth.append(A_xx)

            A_xz = np.random.randint(low=0,high=2,size=(num_X,num_Z)) # z->x
            A_zx = np.zeros(shape=(num_Z,num_X)) # x->z
            A_zz = np.diag( np.random.randint(low=0, high=2, size=num_Z) ) # z->z

            tmp_A = np.concatenate( (
                np.concatenate((A_xx,A_xz),axis=1), 
                np.concatenate((A_zx,A_zz),axis=1) ), axis=0) 
            A.append(tmp_A)
            

            # W
            sigma = np.random.uniform(low=0.01,high=0.1,size=(num_X+num_Z,num_X+num_Z))
            mu = np.random.uniform(low=0.1,high=0.4,size=(num_X+num_Z,num_X+num_Z))
            tmp_W = np.random.normal(loc=mu,scale=sigma)*tmp_A
            W.append(tmp_W)

        return A, W, groudtruth


    @staticmethod         
    def simulate_sem(W, num_X, num_Z, max_lag, num_samples, noise_distribution, num_gaussian_component):
        """ Simulate data for all subjects.
            Inputs:
                matrix: a list with size num_groups, each element is a binary tensor with shape (pl+1,m,m)
            
            In each group k, we consider the following generation:
                With the graph k(i.e. matrix[k]),  we generate data with shape (Ts,m) for each subject, then we have a numpy.array with shape(num_subjects_per_group,Ts,m)
                Note that here DAG[i,j]=1 denotes j->i.
            Returns:
                X: a list with size num_groups, each element is a numpy.array with shape(num_subjects_per_group,Ts,m)
        """

        # Make data stable
        burn_in = 5000
        T = burn_in
        N = num_X+num_Z

        Pi_k_prime = np.random.uniform(low=0.3,high=0.6,size=num_gaussian_component)
        # Make sure the sum of Pi_k_prime is 1
        Pi_k_prime = Pi_k_prime/Pi_k_prime.sum()

        Mu_k_prime = np.random.uniform(low=0.4,high=0.6,size=num_gaussian_component) - np.random.randint(low=0,high=2,size=num_gaussian_component) # Uniform(-0.6,-0.4) U  Uniform(0.4,0.6)

        Sigma_k_prime = np.random.uniform(low=0.2,high=0.5,size=num_gaussian_component) 
        # Generate noise
        if noise_distribution == 'MoG':

            noise = np.zeros(shape=(T,N))

            for i in range(N):
                for t in range(T):     
                    # Mixture of Gaussian
                    k_prime = np.random.choice(range(len(Pi_k_prime)),p=Pi_k_prime)
                    noise[t,i] = np.random.normal(loc=Mu_k_prime[k_prime],scale=Sigma_k_prime[k_prime],size=1)
       
        else:
            raise ValueError('Undefined noise_distribution type')

        data = np.zeros(shape=(T,N))

        # For data(t=1)
        data[0] = noise[0]

        # For data(t=2,...,max_lag)
        for t in range(2,max_lag+1):
            
            tmp_X = np.zeros(num_X)
            tmp_Z = np.zeros(num_Z)
            
            for lag in range(1,t):
                
                coefficient = W[lag-1]

                coefficient_11 = coefficient[0:num_X,0:num_X] # i.e., A_11⊙W_11
                coefficient_12 = coefficient[0:num_X,num_X:N] # i.e., A_12⊙W_12
                coefficient_22 = coefficient[num_X:N,num_X:N] # i.e., A_22⊙W_22

                
                tmp_X +=  (  np.matmul(coefficient_11,data[t-lag-1,0:num_X]) +
                             np.matmul(coefficient_12,data[t-lag-1,num_X:N]) )
                            
                if num_Z:
                    tmp_Z +=   np.matmul(coefficient_22,data[t-lag-1,num_X:N]) 

            data[t-1,0:num_X] = tmp_X + noise[t-1,0:num_X]
            data[t-1,num_X:N] = tmp_Z + noise[t-1,num_X:N]

            
        # For data(t=max_lag+1,T)
        for t in range(max_lag+1, T+1):

            tmp_X = np.zeros(num_X)
            tmp_Z = np.zeros(num_Z)
            
            for lag in range(1,max_lag+1):
                

                coefficient = W[lag-1]

                coefficient_11 = coefficient[0:num_X,0:num_X] # i.e., A_11⊙W_11
                coefficient_12 = coefficient[0:num_X,num_X:N] # i.e., A_12⊙W_12
                coefficient_22 = coefficient[num_X:N,num_X:N] # i.e., A_22⊙W_22

                
                tmp_X +=  (  np.matmul(coefficient_11,data[t-lag-1,0:num_X]) +
                             np.matmul(coefficient_12,data[t-lag-1,num_X:N]) )
                            
                if num_Z:
                    tmp_Z +=   np.matmul(coefficient_22,data[t-lag-1,num_X:N]) 

            data[t-1,0:num_X] = tmp_X + noise[t-1,0:num_X]
            data[t-1,num_X:N] = tmp_Z +noise[t-1,num_X:N]

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
    
    num_X, num_Z,  num_samples, max_lag = 5, 2, 3000, 1 

    dataset = SyntheticDataset(num_X, num_Z,  num_samples, max_lag)

    print(dataset.X.shape)
    print(dataset.W)
    print(dataset.groudtruth)
    plot_timeseries(dataset.X[0:100],'X',True)
    if num_Z:
        plot_timeseries(dataset.Z[0:100],'Z',True) 


