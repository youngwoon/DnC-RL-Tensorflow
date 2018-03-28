ENV_ID='Walker2dForward-v1'
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "x_vel_weight-2/random_steps-10" --max_iters 1000 --timesteps_per_batch 5000 --checkpoint_dir checkpoint_64 --log_dir log_64 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "x_vel_weight-1/random_steps-10" --max_iters 1000 --timesteps_per_batch 5000 --checkpoint_dir checkpoint_64 --log_dir log_64 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "x_vel_weight-0.5/random_steps-10" --max_iters 1000 --timesteps_per_batch 5000 --checkpoint_dir checkpoint_64 --log_dir log_64 &
sleep 10s
tensorboard --logdir log
