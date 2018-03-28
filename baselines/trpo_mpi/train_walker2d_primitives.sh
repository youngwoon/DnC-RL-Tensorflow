ENV_ID=Walker2dForward-v1
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --timesteps_per_batch 5000 --log_dir log/walker_primitives --max_iters 1001 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --timesteps_per_batch 5000 --environment_args "applied_force-5" --log_dir log/walker_primitives_perturb --max_iters 1001&

ENV_ID=Walker2dBackward-v1
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --timesteps_per_batch 5000 --log_dir log/walker_primitives --max_iters 1001 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --timesteps_per_batch 5000 --environment_args "applied_force-5" --log_dir log/walker_primitives_perturb --max_iters 1001 &

ENV_ID=Walker2dToy-v1
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --timesteps_per_batch 5000 --log_dir log/walker_meta/stacked_1 --max_iters 10001 --num_hidden_units 50 --stacked 1 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --timesteps_per_batch 5000 --log_dir log/walker_meta/stacked_4 --max_iters 10001 --num_hidden_units 50 --stacked 4 &
