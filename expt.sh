python TaxiTrajectory.py 0.001 1000 > results/taxi_0.001_cmd.log
python TaxiTrajectory.py 0.01 1000 > results/taxi_0.01_cmd.log
python TaxiTrajectory.py 0.1 1000 > results/taxi_0.1_cmd.log
python TaxiTrajectory.py 1 1000 > results/taxi_1.0_cmd.log

python Baseline.py Taxi 0.001 0.1 0.05
python Baseline.py Taxi 0.01 0.1 0.05
python Baseline.py Taxi 0.1 0.1 0.05
python Baseline.py Taxi 1 0.1 0.05

python Baseline.py ECG 0.25 0.1 0.05
python Baseline.py ECG 0.5 0.1 0.05
python Baseline.py ECG 1 0.1 0.05
python Baseline.py ECG 2 0.1 0.05
python Baseline.py ECG 4 0.1 0.05

python ECG.py 0.25 20 > results/ECG_0.25_20_cmd.log
python ECG.py 0.5 20 > results/ECG_0.5_20_cmd.log
python ECG.py 1 20 > results/ECG_1.0_20_cmd.log
python ECG.py 2 20 > results/ECG_2.0_20_cmd.log
python ECG.py 4 20 > results/ECG_4.0_20_cmd.log
