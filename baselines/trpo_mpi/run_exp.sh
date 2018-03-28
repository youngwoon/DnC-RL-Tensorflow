ENV_ID=Walker2dStandup-v1

mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --max_iters 2001 --timesteps_per_batch 5000 --checkpoint_dir checkpoint/stacked_1/policy_32 --log_dir log/stacked_1/policy_32 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --max_iters 2001 --timesteps_per_batch 5000 --checkpoint_dir checkpoint/stacked_4/policy_32 --log_dir log/stacked_4/policy_32 --stacked_obs 4 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --max_iters 2001 --timesteps_per_batch 5000 --checkpoint_dir checkpoint/stacked_1/policy_64 --log_dir log/stacked_1/policy_64 --num_hidden_units 64 &
mpirun -np 4 python3 run_mujoco.py --env $ENV_ID --max_iters 2001 --timesteps_per_batch 5000 --checkpoint_dir checkpoint/stacked_4/policy_64 --log_dir log/stacked_4/policy_64 --num_hidden_units 64 --stacked_obs 4 &
