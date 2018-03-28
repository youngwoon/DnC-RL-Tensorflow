ENV_ID='Walker2dJump-v1'
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "alive_weight-3/random_steps-10/applied_force-100/x_vel_weight-1/alive_reward-0.5/curb_height-0.7" --max_iters 1500 --timesteps_per_batch 5000 --checkpoint_dir checkpoint_64 --log_dir log_64 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "alive_weight-3/random_steps-10/applied_force-100/x_vel_weight-1/alive_reward-0.5/curb_height-0.6" --max_iters 1500 --timesteps_per_batch 5000 --checkpoint_dir checkpoint_64 --log_dir log_64 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "alive_weight-3/random_steps-10/applied_force-100/x_vel_weight-1/alive_reward-0.5/curb_height-0.5" --max_iters 1500 --timesteps_per_batch 5000 --checkpoint_dir checkpoint_64 --log_dir log_64 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "alive_weight-3/random_steps-10/applied_force-100/x_vel_weight-1/alive_reward-0.5/curb_height-0.4" --max_iters 1500 --timesteps_per_batch 5000 --checkpoint_dir checkpoint_64 --log_dir log_64 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "alive_weight-3/random_steps-10/applied_force-100/x_vel_weight-1/alive_reward-0.5/curb_height-0.45" --max_iters 1500 --timesteps_per_batch 5000 --checkpoint_dir checkpoint_64 --log_dir log_64 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "alive_weight-3/random_steps-10/applied_force-100/x_vel_weight-1/alive_reward-0.5/curb_height-0.55" --max_iters 1500 --timesteps_per_batch 5000 --checkpoint_dir checkpoint_64 --log_dir log_64 &



sleep 10s
tensorboard --logdir log
