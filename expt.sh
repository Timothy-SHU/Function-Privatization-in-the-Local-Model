echo "Preprocessing taxi trajectory data..."
# python PreprocCabData.py > /dev/null
# python SelectCabData.py > /dev/null
for METHOD in "Laplace" "Gaussian"; do
    for EPS in 0.001 0.002 0.005 0.01 0.02 0.05 0.1; do
        echo "Running taxi trajectory privatization with ${METHOD} noise and eps = ${EPS}"
        python TaxiTrajectory.py $METHOD $EPS 1 > results/cmd.log
        mv results/cmd.log results/TaxiTrajectory/taxi_${EPS}/taxi_${METHOD}_${EPS}_cmd.log
        for SAMPLE in 0.1 0.2; do
            for WINDOW in 0.05 0.1; do
                echo "Running taxi trajectory baseline with eps = ${EPS}, sample rate ${SAMPLE}, and window scale ${WINDOW}..."
                python Baseline.py Taxi $METHOD $EPS $SAMPLE $WINDOW
            done
        done
    done
done
echo "Collecting taxi trajecotry result statistics..."
python ExptStats.py Taxi Laplace > results/TaxiTrajectorySummary_GP.txt
python ExptStats.py Taxi Gaussian > results/TaxiTrajectorySummary_CGP.txt

for METHOD in "Laplace" "Gaussian"; do
    for EPS in 0.1 0.2 0.5 1.0 2.0; do
        echo "Running ECG privatization with ${METHOD} noise and eps = ${EPS}"
        python ECG.py $METHOD $EPS 20 > results/cmd.log
        mv results/cmd.log results/ECG/ECG_${EPS}_16x50/ECG_${METHOD}_${EPS}_cmd.log
        for SAMPLE in 0.1 0.2; do
            for WINDOW in 0.05 0.1; do
                echo "Running ECG baseline with eps = ${EPS}, sample rate ${SAMPLE}, and window scale ${WINDOW}..."
                python Baseline.py ECG $METHOD $EPS $SAMPLE $WINDOW
            done
        done
    done
done
echo "Collecting ECG result statistics..."
python ExptStats.py ECG Laplace > results/ECGSummary_GP.txt
python ExptStats.py ECG Gaussian > results/ECGSummary_CGP.txt

python ExptFigs.py

python Synthetic.py
python SyntheticAdapt.py