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

    # parser.add_argument('--sigma_Z',
    #                     type=float,
    #                     default=1.0,
    #                     help='the standard deviation parameter of Normal distribution over latent variables Z')

    # parser.add_argument('--sigma_X',
    #                     type=float,
    #                     default=1.0,
    #                     help='the standard deviation parameter of Normal distribution over observed variables X')
    

# run this script with command: nohup bash run.sh &

for rho in  0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
        
    nohup python -u main.py --prior_rho_A $rho > nohup.txt 2>&1 &
    sleep 0.1
    
done

wait

for sigma_W in  0.005 0.01 0.05 0.1 0.2 0.3 0.5 1.0 2.0 2.5 5.0
do
        
    nohup python -u main.py --prior_sigma_W $sigma_W > nohup.txt 2>&1 &
    sleep 0.1
    
done

wait

for sigma_X in  0.03 0.05 0.08 0.1 0.2 0.3 0.5 0.8 1.0 2.0 2.5 5.0
do
        
    nohup python -u main.py --sigma_X $sigma_X > nohup.txt 2>&1 &
    sleep 0.1
    
done

wait

for sigma_Z in  0.5 1.0 1.5 2.0 2.5 3.0 5.0 
do
        
    nohup python -u main.py --sigma_Z $sigma_Z > nohup.txt 2>&1 &
    sleep 0.1

done

wait

# for seed in  100 200 300 400 500 600 700 800 900 1000
# do
        
#     nohup python -u main.py --seed $seed > nohup.txt 2>&1 &
#     sleep 0.1
    
# done


# for rho in  0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
# do
#     for sigma_X in  0.01 0.005 0.1 0.5 1.0 1.5 2.0 5.0 
#     do
#         for sigma_Z in  0.01 0.05 0.1 0.5 1.0 1.5 2.0 5.0 
#         do
                        
#         nohup python -u main.py --prior_rho_A $rho --sigma_X $sigma_X --sigma_Z $sigma_Z --learning_rate $lr --num_iterations $it > nohup.txt 2>&1 &
#         sleep 0.1
                    
#         done
#         wait
 
#     done

# done
