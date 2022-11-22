# DMDNA_CODE
The code for the Double Layer Multi-robots Path Planner based on Deep Multiple Dueling Network Agent
With its advantages of high efficiency and robustness, Multiple robots system has better application prospects in 
some dynamic and complex environments, such as smart factories. However, as the number of robots increases, 
it is critical to efficiently coordinate congestion and jams among robots. To address this problem, we design the 
Double Layer Multi-robots Path Planner to achieve the coordination of robots conflicts. In the Collaboration Layer, 
we propose a Deep Multiple Dueling Network Agent( DMDNA) node distribution agent based on the idea of forward 
node distribution to coordinate the path conflict problem of multiple robots. In the DMDNA, we design the Multiple 
Dueling Network(MDN) model based on the DDQN network structure to cope with the high dimensional discrete 
action space problem in the multiple robots. In addition, in order to overcome the characteristics of discrete and 
sparse rewards of reinforcement learning, we add the Hindsight Experience Replay(HER) experience replay strategy 
in the training of DMDNA. The experience of primary sequences is cut and reused to improve the utilization of samples. 
In the Motion Layer, we propose Adaptive-DWA(A-DWA) algorithm, which adding the Target Function. 
According to the Target Function, the forward simulation time and the weight of the evaluation function 
can be changed dynamically to improve the efficiency of local path planning and dynamic obstacle avoidance. 
The final experimental results show that the Double Layer Multi-robots Path Planner based on DMDNA has certain 
advantages over traditional reinforcement learning methods in terms of success rate and time cost in solving multiple
robots path planning problems.