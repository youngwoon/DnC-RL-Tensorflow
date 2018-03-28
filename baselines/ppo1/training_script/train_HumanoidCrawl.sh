ENV_ID='HumanoidCrawl-v1'
python3 run_mujoco.py --env $ENV_ID --environment_args "side_weight-0/ceiling_reward-2/x_vel_weight-0.25" --num-timesteps 5000000 & 
python3 run_mujoco.py --env $ENV_ID --environment_args "side_weight-0/ceiling_reward-1/x_vel_weight-0.25" --num-timesteps 5000000 &
python3 run_mujoco.py --env $ENV_ID --environment_args "side_weight-0/ceiling_reward-0.5/x_vel_weight-0.25" --num-timesteps 5000000 &
python3 run_mujoco.py --env $ENV_ID --environment_args "side_weight-0/ceiling_reward-1.5/x_vel_weight-0.25" --num-timesteps 5000000 &
python3 run_mujoco.py --env $ENV_ID --environment_args "side_weight-0.5/ceiling_reward-2.5/x_vel_weight-0.25" --num-timesteps 5000000 &  
sleep 10s
tensorboard --logdir log
