echo "Preprocessing taxi trajectory data..."
python PreprocCabData.py > /dev/null
python SelectCabData.py > /dev/null
for EPS in 0.001 0.003 0.01 0.03 0.1; do
    echo "Runing taxi trajectory privatization with eps = ${EPS}"
    python TaxiTrajectory.py $EPS 1 > results/taxi_${EPS}_cmd.log
    mv results/taxi_${EPS}_cmd.log results/TaxiTrajectory/taxi_${EPS}/taxi_${EPS}_cmd.log
    for SAMPLE in 0.1 0.2; do
        for WINDOW in 0.05 0.1; do
            echo "Running taxi trajectory baseline with eps = ${EPS}, sample rate ${SAMPLE}, and window scale ${WINDOW}..."
            python Baseline.py Taxi $EPS $SAMPLE $WINDOW
        done
    done
done
echo "Collecting taxi trajecotry result statistics..."
python ExptStats.py Taxi > results/TaxiTrajectorySummary.txt

for EPS in 0.25 0.5 1.0 2.0 4.0; do
    echo "Runing ECG privatization with eps = ${EPS}"
    python ECG.py $EPS 20 > results/ECG_${EPS}_20_cmd.log
    mv results/ECG_${EPS}_20_cmd.log results/ECG/ECG_${EPS}_16x50/ECG_${EPS}_20_cmd.log
    for SAMPLE in 0.1 0.2; do
        for WINDOW in 0.05 0.1; do
            echo "Running ECG baseline with eps = ${EPS}, sample rate ${SAMPLE}, and window scale ${WINDOW}..."
            python Baseline.py ECG $EPS $SAMPLE $WINDOW
        done
    done
done
echo "Collecting ECG result statistics..."
python ExptStats.py ECG > results/ECGSummary.txt
