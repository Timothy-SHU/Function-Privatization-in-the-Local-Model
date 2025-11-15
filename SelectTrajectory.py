from PrivPwcApprox import *
from AdaptApprox import *
from datetime import datetime

timer = time.time()
df = pd.read_pickle("cabspottingdata/trajectory.pkl")
print("="*50)
print(f"Datasets loaded in {time.time()-timer:.5f} sec.")
print(f"Total # of datapoints: {np.sum([len(df['t'][i]) for i in range(len(df))])}")
print("="*50)
start_t = min(df['t'].apply(min))
end_t = max(df['t'].apply(max))
start_t = datetime(2008, 5, 25, 0, 0)
end_t = datetime(2008, 5, 30, 0, 0)
print(start_t, end_t)
in_interval = np.vectorize(lambda t: start_t < t < end_t)
count = df['t'].apply(in_interval).apply(sum).tolist()
print(min(count), max(count), sum(count))
print(count)
print()