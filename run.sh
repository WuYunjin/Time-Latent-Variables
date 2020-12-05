#!/bin/bash
    # ##### Model settings #####
    # parser.add_argument('--prior_rho_A',
    #                     type=float,
    #                     default=0.7,
    #                     help='the parameter of Bernoulli distribution, which is the prior over A_k,ij')

    # parser.add_argument('--prior_sigma_W',
    #                     type=float,
    #                     default=1.0,
    #                     help='the standard deviation parameter of Normal distribution, which is the prior over W_k,ij')

    # parser.add_argument('--temperature',
    #                     type=float,
    #                     default=2.0,
    #                     help='the temperature parameter of Gumbel-Bernoulli distribution')
    
    # parser.add_argument('--sigma_Z',
    #                     type=float,
    #                     default=1.0,
    #                     help='the standard deviation parameter of Normal distribution over latent variables Z')

    # parser.add_argument('--sigma_X',
    #                     type=float,
    #                     default=1.0,
    #                     help='the standard deviation parameter of Normal distribution over observed variables X')
    
    # ##### Training settings #####
    # parser.add_argument('--learning_rate',
    #                     type=float,
    #                     default=1e-3,
    #                     help='Learning rate for optimizer') # sensitive


    # parser.add_argument('--num_iterations',
    #                     type=int,
    #                     default=2000,
    #                     help='Number of iterations')

    # parser.add_argument('--num_output',
    #                     type=int,
    #                     default=10,
    #                     help='Number of iterations to display information')

# run this script with command: nohup bash run.sh &

for rho in  0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    for sigma_X in  0.01 0.005 0.1 0.5 1.0 1.5 2.0 5.0 
    do
        for sigma_Z in  0.01 0.05 0.1 0.5 1.0 1.5 2.0 5.0 
        do

            for lr in  5e-3 1e-3 1e-4 1e-5
            do
                    
                for it in 1250 1500 1750 2000 2500
                do
                        
                    nohup python -u main.py --prior_rho_A $rho --sigma_X $sigma_X --sigma_Z $sigma_Z --learning_rate $lr --num_iterations $it > nohup.txt 2>&1 &
                    sleep 0.1
                    
                done
                wait
 
            done

        done
    
    done

done














# use 0.6
# for rho in  0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
# do
        
#     nohup python -u main.py --prior_rho_A $rho > nohup.txt 2>&1 &
    
# done


# for sigma_X in  0.01 0.1 0.5 1.0 1.5 2.0 2.5 
# do
        
#     nohup python -u main.py --sigma_X $sigma_X > nohup.txt 2>&1 &
#     sleep 0.1
    
# done

# wait

# for sigma_Z in  0.01 0.1 0.5 1.0 1.5 2.0 2.5 
# do
        
#     nohup python -u main.py --sigma_Z $sigma_Z > nohup.txt 2>&1 &
#     sleep 0.1

# done

# wait

# for t in  0.1 0.5 1.0 2.0 5.0
# do
        
#     nohup python -u main.py --temperature $t > nohup.txt 2>&1 &
#     sleep 0.1
    
# done

# wait

# for lr in  5e-3 1e-3 1e-4 1e-5
# do
        
#     nohup python -u main.py --learning_rate $lr > nohup.txt 2>&1 &
#     sleep 0.1
    
# done

# wait

# for it in 1250 1500 1750 2000 2500
# do
        
#     nohup python -u main.py --num_iterations $it > nohup.txt 2>&1 &
#     sleep 0.1
    
# done
