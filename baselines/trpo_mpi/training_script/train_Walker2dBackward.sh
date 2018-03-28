ENV_ID='Walker2dBackward-v1'
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "x_vel_weight-1" --max_iters 1000 --timesteps_per_batch 5000 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "x_vel_weight-2" --max_iters 1000 --timesteps_per_batch 5000 &
sleep 10s
tensorboard --logdir log
