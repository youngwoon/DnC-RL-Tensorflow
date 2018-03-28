ENV_ID='HumanoidBackward-v1'
python3 run_mujoco.py --env $ENV_ID --environment_args "side_weight-0/direction_weight-0/alive_weight-5/x_vel_weight-0.25" --num-timesteps 5000000 & 
python3 run_mujoco.py --env $ENV_ID --environment_args "side_weight-0/direction_weight-1/alive_weight-5/x_vel_weight-0.25" --num-timesteps 5000000 &
python3 run_mujoco.py --env $ENV_ID --environment_args "side_weight-0/direction_weight-2/alive_weight-5/x_vel_weight-0.25" --num-timesteps 5000000 &
python3 run_mujoco.py --env $ENV_ID --environment_args "side_weight-0/direction_weight-3/alive_weight-5/x_vel_weight-0.25" --num-timesteps 5000000 &
python3 run_mujoco.py --env $ENV_ID --environment_args "side_weight-0.5/direction_weight-0/alive_weight-5/x_vel_weight-0.25" --num-timesteps 5000000 &  
python3 run_mujoco.py --env $ENV_ID --environment_args "side_weight-0.5/direction_weight-1/alive_weight-5/x_vel_weight-0.25" --num-timesteps 5000000 &
python3 run_mujoco.py --env $ENV_ID --environment_args "side_weight-0.5/direction_weight-2/alive_weight-5/x_vel_weight-0.25" --num-timesteps 5000000 &
python3 run_mujoco.py --env $ENV_ID --environment_args "side_weight-0.5/direction_weight-3/alive_weight-5/x_vel_weight-0.25" --num-timesteps 5000000 &
sleep 10s
tensorboard --logdir log
