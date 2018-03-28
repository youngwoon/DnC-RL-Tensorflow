ENV_ID='HumanoidCrawl-v1'
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "side_weight-0/ceiling_reward-1/x_vel_weight-0.25" --max_iters 1000 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "side_weight-0/ceiling_reward-2/x_vel_weight-0.25" --max_iters 1000 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "side_weight-0/ceiling_reward-3/x_vel_weight-0.25" --max_iters 1000 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "side_weight-1/ceiling_reward-1/x_vel_weight-0.25" --max_iters 1000 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "side_weight-1/ceiling_reward-2/x_vel_weight-0.25" --max_iters 1000 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "side_weight-1/ceiling_reward-3/x_vel_weight-0.25" --max_iters 1000 &
sleep 10s
tensorboard --logdir log
