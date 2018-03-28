ENV_ID='Walker2dWalkLower-v1'
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "height_wieght-0.5" --max_iters 1000 --timesteps_per_batch 5000 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "height_wieght-1" --max_iters 1000 --timesteps_per_batch 5000 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --environment_args "height_wieght-2" --max_iters 1000 --timesteps_per_batch 5000 &
sleep 10s
tensorboard --logdir log
