ENV_ID='MetaWalker2d2-v1'
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "x_vel_weight-1/time_penalty-0.1" --max_iters 2000 --timesteps_per_batch 5000 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "x_vel_weight-1/time_penalty-0.5" --max_iters 2000 --timesteps_per_batch 5000 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "x_vel_weight-0.5/time_penalty-0.1" --max_iters 2000 --timesteps_per_batch 5000 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "x_vel_weight-0.5/time_penalty-0.5" --max_iters 2000 --timesteps_per_batch 5000 &
sleep 10s
tensorboard --logdir log
