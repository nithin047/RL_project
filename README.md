# A Reinforcement Learning Approach to Managing Association and Handovers in Cellular Networks
Authors: Saadallah Kassir, Nithin S. Ramesan

Dependencies: gym, matplotlib, numpy, torch, tqdm, scipy, math

## Switching between phases of projects (change parameters wherever an instance of the environment is initialized.):
### Phase 1:
Set the parameter lambdaUE = 0, handoffDuration = 0, velocity = 0

### Phase 2: 
Set the parameter handoffDuration = 0, velocity = 0

### Phase 3: 
Leave parameters as is.

## Running algorithms

### REINFORCE
Use the command `python3 reinforce.py model-name`, where model-name is the name you'd like the trained policy network to be saved under. Use the extension .pt. Parameters for REINFORCE can be changed in reinforce.py

### PPO
Navigate into the following path: ./gym_algo/gym-NetworkEnvironment and run `pip install -e .`
Then navigate back to ./gym_algo/ and run `python3 main.py --env-name "gym_NetworkEnvironment:NetworkEnvironment-v3" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 2000000 --use-linear-lr-decay --use-proper-time-limits`, with values of parameters changed if necessary. The trained model will be saved in ./gym_algo/trained_models.

## Evaluating Algorithms

Run any of the evaluation scripts using the command format `python3 evaluation_script_name.py model_name.pt`, while ensuring that the model file is present in the same folder - you will have to move the PPO trained model from the above mentioned folder into this one.

## General Notes
* To train Phase 1 and Phase 2 successfully, you will need to permute the SINR vectors (see report for why). 
* While evaluating rate for phase 3, the RL policy could return zero rate for short runs of the evaluation script due to the handovers. Run the evaluation script for more iterations if this occurs.
