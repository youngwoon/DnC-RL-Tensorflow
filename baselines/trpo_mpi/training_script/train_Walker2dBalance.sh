ENV_ID='Walker2dBalance-v1'
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "tolerance-0.2/balance_penalty-1.5" --timesteps_per_batch 5000 --max_iters 1000 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "tolerance-0.1/balance_penalty-1.5" --timesteps_per_batch 5000 --max_iters 1000 &
sleep 10s
tensorboard --logdir log
