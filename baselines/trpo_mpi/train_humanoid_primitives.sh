ENV_ID=HumanoidForward-v1
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --timesteps_per_batch 5000 --log_dir log/humanoid_primitives --max_iters 3001 --environment_args "x_vel_weight-5" --num_hidden_units 100 --prefix units.100 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --timesteps_per_batch 5000 --log_dir log/humanoid_primitives --max_iters 3001 --environment_args "x_vel_weight-5" --num_hidden_units 200 --prefix units.200 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --timesteps_per_batch 5000 --log_dir log/humanoid_primitives --max_iters 3001 --environment_args "x_vel_weight-5" --num_hidden_units 300 --prefix units.300 &

#mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --timesteps_per_batch 5000 --log_dir log/humanoid_primitives_perturb --max_iters 3001 --environment_args "x_vel_weight-5/applied_force-5" &

ENV_ID=HumanoidBackward-v1
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --timesteps_per_batch 5000 --log_dir log/humanoid_primitives --max_iters 3001 --environment_args "x_vel_weight-5" --num_hidden_units 100 --prefix units.100 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --timesteps_per_batch 5000 --log_dir log/humanoid_primitives --max_iters 3001 --environment_args "x_vel_weight-5" --num_hidden_units 200 --prefix units.200 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --timesteps_per_batch 5000 --log_dir log/humanoid_primitives --max_iters 3001 --environment_args "x_vel_weight-5" --num_hidden_units 300 --prefix units.300 &


#mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --timesteps_per_batch 5000 --log_dir log/humanoid_primitives_perturb --max_iters 3001 --environment_args "x_vel_weight-5/applied_force-5" &

ENV_ID=HumanoidHurdle-v1
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --timesteps_per_batch 10000 --log_dir log/humanoid_primitives --max_iters 10001 --num_hidden_units 300 --prefix units.300 &

ENV_ID=HumanoidJump-v1
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --timesteps_per_batch 10000 --log_dir log/humanoid_primitives --max_iters 10001 --num_hidden_units 300 --prefix units.300 &
