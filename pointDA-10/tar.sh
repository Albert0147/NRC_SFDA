python train_tar.py -source scannet -target modelnet -scaler 1 -weight 1 -tb_log_dir ./logs/sa_ss2m -lr 1e-5 -K 2 -KK 2 #

python train_tar.py -source scannet -target shapenet -scaler 1 -weight 0.5 -tb_log_dir ./logs/sa_ss2s -lr 1e-6 -K 4 -KK 3 #

python train_tar.py -source modelnet -target scannet  -scaler 1 -weight 0.5 -tb_log_dir ./logs/sa_m2ss -lr 1e-6 -K 5 -KK 5 # 

python train_tar.py -source modelnet -target shapenet -scaler 1 -weight 1 -tb_log_dir ./logs/sa_m2s -lr 1e-6 -K 5 -KK 5 # 

python train_tar.py -source shapenet -target modelnet -scaler 1 -weight 1 -tb_log_dir ./logs/sa_s2m -gpu 1 -K 3 -KK 2 # 

python train_tar.py -source shapenet -target scannet -scaler 1 -weight 0.5 -tb_log_dir ./logs/sa_s2ss -lr 1e-6 -K 4 -KK 4 #
