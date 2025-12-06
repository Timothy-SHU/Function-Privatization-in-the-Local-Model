# Function Privatization under GP

This repository implements function privatization under GP metrics.

The essential utilities are implemented in `PrivPwcApprox.py`, where the class $\texttt{PrivatePiecewiseApprox}$ encapsulates the functional least-squares solver along with the privatization mechanism.

`AdaptiveApprox.py` implements the adaptive privatization mechanisms $\texttt{PrivFuncSeg}$ and $\texttt{ReduceSeg}$ mentioned in the paper. The detailed runtime figures and decisions are recorded in `info.log` during execution.

## Implementation Details

A solver can be created with `PrivatePiecewiseApprox(interval, breakpoints, basis_type, degree, parallel)`, where the Boolean parameter `parallel` indicates whether enabling parallelism in preprocessing phase.
Supported basis include:

- $\texttt{Polynomial}$: polynomial basis, with degree specified in `degree`. When degree is at most 2, the basis generated is orthonormal.
- $\texttt{Linear-2D}$: 2D linear basis equivalent to $\phi_1(t)=(1,0)$, $\phi_2(t)=(t,0)$, $\phi_3(t)=(0,1)$, and $\phi_4(t)=(0,t)$; converted into orthonormal basis by default.
- $\texttt{Fourier}$: partial Fourier basis $\phi_0(x)=1$ and $\phi_k(x)=\sin(2k\pi x),\phi_k'(x)=\cos(2k\pi x)$ for $k=1,2,\cdots,\text{degree}$.
- $\texttt{Sinc}$: bounded sinc function $\phi_k(x)=\sin(\pi(x-k))/(\pi(x-k))$ for $k=1,2,\cdots,\text{degree}$., where the degree is also known as the shift; note that it has no value (0) outside `interval`.
- $\texttt{Sinc-unbounded}$: unbounded sinc function similar to above, but has value everywhere on $\mathbb{R}$; must be orthonormal.

Use `fit(func)` to obtain least-squares functional approximation of a function `func`. For 2D function, to speed-up computation, convert `func` further into array `func_2D` of two scalar functions and call `fit(func, func_2D)`. The additional `parallel` parameter specifies whether enabling parallelism during approximation. The resulting approximation can be constructed using `createApprox()` method.

To privatize the approximation, call `privatize(epsilon, method)`, where the mechanisms supported are `Laplace` and `Gaussian`.  The Laplace mechanism achieves $\varepsilon$-GP, while the Gaussian mechanism achieves $\varepsilon$-CGP. The privatized function can be obtained with `createPriv()`.

The $l_2$-distance from the original function to the approximation and privatization can be measured through `eval(type)`. `type = 'Approx'` returns $||f-f_\text{approx}||$, and `type = 'Priv'` returns $||f-f_\text{priv}||$. Another method `evalPrivLoss()` measures $||f_\text{approx}-f_\text{priv}||$.

The adaptive approximation is called with `adaptive_approx(func, interval, basis, degree, epsilon, beta, method, func_2D, SVT_threshold_scale, parallel)`. Parameter `SVT_threshold_scale` controls the accuracy of the approximation, with the trade-off in execution time. All other parameters has the same meaning as described above.

## Examples and Empirical Datasets

`Gaussian.py` privides the example of privatizing the sum of two Gaussian curves. The basis used is the degree-3 polynomial basis and the privacy quota is $\varepsilon=0.5$.

`Trigonometric.py` privides the example of privatizing the sum of two trigonometric functions. The basis used is the partial Fourier basis with degree 2 and the privacy quota is $\varepsilon=0.5$.

The taxi trajectory dataset is under `cabspottingdata/` directory. `PreprocCabData.py` preprocesses the data by converting latitude and longitude into coordinates (with meter as unit) and removing erroneous datapoints. `SelectCabData.py` selects curves whose time range contains 8:00 ~ 20:00 everyday, and filters out those with less than 500 samples. `TaxiTrajectory.py` privatizes the selected curves with 2D linear basis, and record the $l_2$-losses.

Directly calling `TaxiTrajectory.py` will execute in interactive mode, where one sample curve of 1000 points is privatized with $\varepsilon=0.01$.

For batch experiment, add 3 arguments when calling the script: the privacy quota, the batch size, and the SVT threshold factor. Each curve is privatized and recorded for 20 times. The results will be stored under `results/TaxiTrajectory/taxi_{args}/` directory.
```
python PreprocCabData.py
python SelectCabData.py
python TaxiTrajectory.py 0.01 1 > results/taxi_cmd.log
```

The ECG dataset is under `ptb-xl/` directory. `ECG.py` loads the records and privatizes them with bounded Sinc basis. Here we use public info of average QRS interval length and scale the time by a factor of 80. Each ECG record consists of 1000 datapoints sampled at frequency 10 Hz, so each record spans 10 seconds (thus 800 after scaling). We split it to $m$ intervals, where each interval has bounded Sinc basis with shift $800/m$.

Directly calling `ECG.py` will execute in interactive mode with $\varepsilon=1$, where the bounded Sinc basis (if used) has $m$ set to 100.

For batch experiment, add 2 arguments when calling the script: the privacy quota and the batch size. When the batch size inputted is -1, unbounded Sinc basis will be applied; otherwise, corresponding bounded Sinc basis is used: given batch size $s\in\mathbb{N}^*$, the corresponding $m$ is $m=1000/s$. Each curve is privatized and recorded for 20 times. The results will be stored under `results/ECG/ECG_{args}/` directory.
```
python ECG.py 1.0 5 > results/ECG_cmd.log
python ECG.py 1.0 5 > results/ECG_cmd.log
```
The first command splits into 200 intervals, where each interval is equipped with Sinc basis of shift 5.
The second command splits into 50 intervals, where each interval is equipped with Sinc basis of shift 16.

We added efficient integration for $\texttt{Linear-2D}$ basis and $\texttt{Sinc}$ basis based on purely arithmetic computations, so multiprocessing is not needed for these bases. For other bases, integration uses `scipy.integrate.quad`, and enabling multiprocessing is recommended.

Baseline is implemented in `Baseline.py`. It accepts three arguments: `Taxi` or `ECG` to specify the target dataset, a sample rate, and a window scale. The number of sample datapoints drawn is $(\texttt{sample rate})\times(\texttt{total num of datapoints})$, and the window size is $(\texttt{window scale})\times(\texttt{num of samples})$.

After the experiment, use `ExptStats.py` with argument `Taxi` or `ECG` to generate summary statistics table for the datasets.

The shell script `expt.sh` runs all experiments with our algorithm and baseline, then generate all result statistics. 

## Experiment Setup

### Taxi Trajectory Dataset

**Preprocessing**

- There are 536 trajectories in the dataset, each with the mobility information of a specific cab during a time period of 20 to 40 days.
- We first split the trajectories to curves by date (1 curve represents 1 cab's data in 1 day), and select the observation window "8:00 ~ 20:00". Any curve that does not fully cover this window is removed.
- We truncate each curve to this window with interpolation at endpoints: fix endpoints at 8:00 and 20:00, decide the coordinates at the endpoints by interpolating the two adjacent datapoints. Thus each curve starts exactly at 8:00 and ends exactly at 20:00, spaning 12 hours = 43200 seconds.
- To ensure sample size, after truncating each curve to 8:00 ~ 20:00, we remove all curves with less than 500 datapoints.
- There are 5102 curves remaining, each with 500 to 850 datapoints.

**Experiment**

- Our algorithm include three phases: (1) adaptive least-squares approximation (with SVT threshold exactly $\tau_j=2^j/(\epsilon/4)$), (2) privatize by adding noises to the coefficients, and (3) smooth (ensure continuity) by solving a QP.
- The baseline has two parameters, $\texttt{sample rate}$ and $\texttt{window scale}$, and has two phases: (1) add Laplace noises to the sample and directly connect these noised samples to form privatized curve, and (2) take window average of the noised samples, and connect those "smoothed" samples to form a privatized curve.
- For each choice of $\varepsilon=0.001, 0.003, 0.01, 0.03, 0.1$, run our algorithm ((1)+(2)+(3)) on each curve for 20 times, and record approximation error, privatization error, and smoothed privatization error.
- For each choice of $\varepsilon=0.001, 0.003, 0.01, 0.03, 0.1$ and all 4 combinations of $\texttt{sample rate}\in\{0.1,0.2\}$ and $\texttt{window scale}=\{0.05,0.1\}$,  run the baseline ((1)+(2)) for 20 times on each curve and record errors.

**Evaluation**

For a specific curve, for each algorithm, remove the 4 records with 2 largest and 2 smallest privatization error (so 20% records removed), then take average of the remaining 16 records. The average errors are considered as the empirical mean utility loss aquired by this algorithm on this curve.

For each algorithm, take average of its average errors aquired on all curves in the dataset, and report the empirical mean utility loss of it on the whole dataset.

### ECG Dataset

**Preprocessing**

- There are 21799 records in the dataset, each represents the ECG record of 10 seconds. The record consists of 12 curves, and we only consider the second curve, ECG lead II, as it is the most representative one (so each record contains only 1 curve).
- Each of those records contains 1000 datapoints, sampleded at 100 Hz.
- The time range of each record is 10 seconds, and we multiply the time by 80 (time range becomes 800 units now) to match the QRS interval with the shape of standard sinc function.
- The amplitude of the records are measured in millivolt (mV), and we converted it to microvolt ($\mu$V).

**Experiment**

- The choices of $\varepsilon$ are $0.25, 0.5, 1.0, 2.0, 4.0$.
- Our algorithm include three phases: (1) piecewise least-squares approximation with 50 pieces, where each piece is fitted with basis $\{\text{sinc}(t-k)\}_{k=0}^{15}$, (2) privatize by adding noises to the coefficients, and (3) smooth (ensure continuity) by solving a QP.
- The baseline and its experiment are exactly the same as those above.
- The experiment on our algorithm is almost identical to that above, except that we only run (1) once and run (2)+(3) for 20 times. This is valid because the least-squares approximation result remains the same with the same set of breakpoints and basis.

**Evaluation**

Identical to that above.
