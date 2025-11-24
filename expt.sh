python TaxiTrajectory.py 0.01 1000 10 n > results/taxi_cmd.log
python ECG.py 1 20 n n > results/ECG_cmd.log
python Baseline.py ECG 1 12000
python Baseline.py Taxi 0.01 400000

