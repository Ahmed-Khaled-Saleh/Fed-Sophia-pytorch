
#!/bin/bash
python run_all.py --dataset Mnist --model MLP --algorithm Sophia --learning_rate 0.005  --tau 10 --L 0.001 --num_global_iters 100 --local_epochs 1
python run_all.py --dataset Mnist --model MLP --algorithm Sophia --learning_rate 0.01  --tau 10 --L 0.001 --num_global_iters 100 --local_epochs 1 
python run_all.py --dataset Mnist --model MLP --algorithm Sophia --learning_rate 0.1  --tau 10 --L 0.001 --num_global_iters 100 --local_epochs 1 

python run_all.py --dataset Mnist --model MLP --algorithm Sophia --learning_rate 0.005  --tau 5 --L 0.001 --num_global_iters 100 --local_epochs 1
python run_all.py --dataset Mnist --model MLP --algorithm Sophia --learning_rate 0.01  --tau 5 --L 0.001 --num_global_iters 100 --local_epochs 1
python run_all.py --dataset Mnist --model MLP --algorithm Sophia --learning_rate 0.1  --tau 5 --L 0.001 --num_global_iters 100 --local_epochs 1

python run_all.py --dataset Mnist --model MLP --algorithm Sophia --learning_rate 0.005  --tau 1 --L 0.001 --num_global_iters 100 --local_epochs 1
python run_all.py --dataset Mnist --model MLP --algorithm Sophia --learning_rate 0.01  --tau 1 --L 0.001 --num_global_iters 100 --local_epochs 1
python run_all.py --dataset Mnist --model MLP --algorithm Sophia --learning_rate 0.1  --tau 1 --L 0.001 --num_global_iters 100 --local_epochs 1

python run_all.py --dataset Mnist --model MLP --algorithm Sophia --learning_rate 0.1  --tau 1 --L 0.001 --num_global_iters 100 --local_epochs 5
python run_all.py --dataset Mnist --model MLP --algorithm Sophia --learning_rate 0.1  --tau 1 --L 0.001 --num_global_iters 100 --local_epochs 5
python run_all.py --dataset Mnist --model MLP --algorithm Sophia --learning_rate 0.1  --tau 1 --L 0.001 --num_global_iters 100 --local_epochs 5






================================================================================================================================================================
# Experiment 1
python run_all.py --dataset Mnist --model mclr --algorithm DONE --batch_size 0 --alpha 0.03 --num_global_iters 100 --local_epochs 40 --numedges 32

# Experiment 2
python run_all.py --dataset Mnist --model mclr --algorithm Newton --batch_size 0 --alpha 0.03 --num_global_iters 100 --local_epochs 40 --numedges 32

# Experiment 3
python run_all.py --dataset Mnist --model mclr --algorithm Newton --batch_size 0 --alpha 0.05 --num_global_iters 100 --local_epochs 40 --numedges 32

# Experiment 4
python run_all.py --dataset Mnist --model mclr --algorithm DANE --batch_size 0 --eta 1 --learning_rate 0.04 --num_global_iters 100 --local_epochs 40 --numedges 32

# Experiment 5
python run_all.py --dataset Mnist --model mclr --algorithm FEDL --batch_size 0 --eta 1 --learning_rate 0.04 --num_global_iters 100 --local_epochs 40 --numedges 32

# Experiment 6
python run_all.py --dataset Mnist --model mclr --algorithm GD --batch_size 0 --learning_rate 0.2 --num_global_iters 100 --numedges 32

# Experiment 7
python run_all.py --dataset Mnist --model mclr --algorithm GT --batch_size 0 --alpha 0.03 --L 1 --num_global_iters 100 --local_epochs 40 --numedges 32
