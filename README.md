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

Baseline is implemented in `Baseline.py`. It accepts three arguments: `Taxi` or `ECG` to specify the target dataset, a sample rate, and a window scale. The number of sample datapoints drawn is $(\texttt{sample rate})\times(\texttt{total \# of datapoints})$, and the window size is $(\texttt{window scale})\times(\texttt{\# of samples})$.

After the experiment, use `ExptStats.py` with argument `Taxi` or `ECG` to generate summary statistics table for the datasets.

The shell script `expt.sh` runs all experiments with our algorithm and baseline, then generate all result statistics. 