ENV_ID='HumanoidForward-v1'
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "side_weight-0/direction_weight-0/alive_weight-5/x_vel_weight-0.25" --max_iters 1000 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "side_weight-1/direction_weight-0/alive_weight-5/x_vel_weight-0.25" --max_iters 1000 &
sleep 10s
tensorboard --logdir log
