# Function Privatization in the Local Model

This repository implements function privatization under geo-privacy metrics, and conducts empirical experiments on synthetic Gaussian curves, [CRAWDAD cab mobility dataset](https://ieee-dataport.org/open-access/crawdad-epflmobility), [PTB-XL ECG dataset](https://physionet.org/content/ptb-xl/1.0.3/). The implementation covers all mechanisms mentioned in the paper, including $\texttt{Project-and-Privatize}$, $`\texttt{PrivFuncSelect}`$, $\texttt{PrivFuncSeg}$, and baseline methods.

## Quick Start

**Dependencies.** We recommend Python 3.9+. Core functionality requires NumPy (≥ 1.21.0) for basic computations, SciPy (≥ 1.7.0) for numerical integration, CVXPY (≥ 1.7.3) for solving QPs, and pathos (≥ 0.3.4) for multiprocessing. Auxiliary functionality requires Matplotlib, pandas, and tqdm.

**Downloading Datasets.** Please download the [CRAWDAD cab mobility dataset](https://ieee-dataport.org/open-access/crawdad-epflmobility) and decompress everything into `cabspottingdata/`. For the [PTB-XL ECG dataset](https://physionet.org/content/ptb-xl/1.0.3/), please download the `records100/00000/` directory of the dataset into `ptb-xl/records100/00000/`, and it is recommended to be done via AWS CLI.

```
tar -xvzf cabspottingdata.tar.gz cabspottingdata/
aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/records100/00000/ ptb-xl/records100/00000/
```

**Running the Full Experiment.** A shell script `expt.sh` is provided to run all experiments. The whole process normally takes 12 to 18 hours. Upon completion, the results will be stored in corresponding folders under `results/`. The numerical summaries will be generated under sub-folders, and the figures will be plotted under `results/figs/`.

**Our Experiment Results.** The results we used in our paper are uploaded to this repository under `results/`. This includes the raw output (archived as Zip files) and the figures generated (in corresponding folders under `results/figs/`).

## Core Functionality Overview

### Project-and-Privatize

This core functionality is implemented in `PrivPwcApprox.py`, where the class $\texttt{PrivatePiecewiseApprox}$ encapsulates the functional least-squares solver along with the privatization mechanism.

A solver can be created with `solver = PrivatePiecewiseApprox(interval, breakpoints, basis_type, degree, parallel)`, where the Boolean parameter $\texttt{parallel}$ indicates whether enabling parallelism in preprocessing phase.
Supported basis include:

- $\texttt{Polynomial}$: polynomial basis, with degree specified in $\texttt{degree}$. When degree is at most 2, the basis generated is orthonormal.
- $\texttt{Linear-2D}$: 2D linear basis equivalent to $\phi_1(t)=(1,0)$, $\phi_2(t)=(t,0)$, $\phi_3(t)=(0,1)$, and $\phi_4(t)=(0,t)$; converted into orthonormal basis by default.
- $\texttt{Fourier}$: partial Fourier basis $\phi_0(x)=1$ and $\phi_k(x)=\sin(2k\pi x),\phi_k'(x)=\cos(2k\pi x)$ for $k=1,2,\cdots,\texttt{degree}$.
- $\texttt{Sinc}$: bounded sinc function $\phi_k(x)=\sin(\pi(x-k))/(\pi(x-k))$ for $k=1,2,\cdots,\texttt{degree}$, where the degree is also known as the shift; note that it has no value (0) outside $\texttt{interval}$.
- $\texttt{Sinc-unbounded}$: unbounded sinc function similar to above, but has value everywhere on $\mathbb{R}$; must be orthonormal.

The $\texttt{breakpoints}$ parameter above must be a sorted list, with the two values in $\texttt{interval}$ as its endpoints. On each sub-interval separated by these breakpoints, a set of basis specified above will be created. Note that $\texttt{degree}$ may also be a sorted list, and the basis created will only include those degrees specified in the list.

A least-squares functional approximation of a function $\texttt{func}$ can be obtained via `solver.fit(func)`. For 2D function, to speed-up computation, we may convert $\texttt{func}$ further into array $\texttt{func}\_\texttt{2D}$ of two scalar functions and call `solver.fit(func, func_2D)`. An additional $\texttt{parallel}$ parameter can be added to specify whether enabling parallelism during approximation. The resulting approximation can be constructed via `solver.createApprox()`.

To privatize the approximation, call `solver.privatize(eps, method)`, where the mechanisms supported are $\texttt{'Laplace'}$ and $\texttt{'Gaussian'}$.  The Laplace mechanism achieves $\varepsilon$-GP, while the Gaussian mechanism achieves $\varepsilon$-CGP. The privatized function can be obtained with `solver.createPriv()`.

The $L_2$-distance from the original function to the approximation and privatization can be measured through `solver.eval(type)`. `type = 'Approx'` returns $||f-f_\text{approx}||$, and `type = 'Priv'` returns $||f-f_\text{priv}||$. Another method `solver.evalPrivLoss()` measures $||f_\text{approx}-f_\text{priv}||$.

To guarantee continuity of privatized curve, call `solver.smooth(method)`. Here the $\texttt{method}$ parameter can be either $\texttt{'CVXPY'}$ or $\texttt{'Sparse-KKT'}$. The former uses CVXPY to solve the QP, and the latter uses the sparse KKT method to numerical solve the QP. Normally, the latter would be more efficient and accurate. Note that this $\texttt{smooth}$ method will *overwrite* the noise generated by the previous $\texttt{privatize}$ method! Also, to measure $||f-f_\text{priv-cts}||$ or $||f_\text{approx}-f_\text{priv-cts}||$, call `solver.eval('Priv')` or `solver.evalPrivLoss()` *after* calling the $\texttt{smooth}$ method.

We added efficient integration for $\texttt{Linear-2D}$ basis and $\texttt{Sinc}$ basis based on purely arithmetic computations, so multiprocessing is not needed for these bases. For other bases, integration uses `scipy.integrate.quad`, and enabling multiprocessing is recommended.

### PrivFuncSelect and PrivFuncSeg

The adaptive basis selection mechanism $`\texttt{PrivFuncSelect}`$ is implemented in `AdaptiveBasis.py`. To apply it on a function $\texttt{func}$, call `adaptive_basis(func, interval, basis, eps, method)`, which returns a $\texttt{PrivatePiecewiseApprox}$ solver and a degree list. The solver returned already generates noises for privatization, but is not guaranteed to be continuous. The degree list returned indicates the basis selected.

`AdaptiveApprox.py` implements the adaptive segmentation mechanism $\texttt{PrivFuncSeg}$ and its subroutine $\texttt{ReduceSeg}$ mentioned in the paper. The adaptive approximation is called with `adaptive_approx(func, interval, basis, degree, eps, beta, method)`, which returns a $\texttt{PrivatePiecewiseApprox}$ solver and remaining budget $B$. Note that the solver returned has **no** noise generated for privatization, meaning that we need to call `solver.privatize(eps = B, method)` immediately afterwards to privatize with privacy budget $B$.

For both adaptive mechanisms, the detailed runtime decisions of SVTs are recorded in `info.log` during execution.

## Datasets and Experiment Scripts

### Synthetic Gaussian Curves

The experiment on synthetic Gaussian curves is implmented in `Synthetic.py` and `SyntheticAdapt.py`. Both script will randomly generate the same 50 synthetic Gaussianc curves with seed 42, where each curve is a linear combination of 1 to 5 random Gaussian curves on intrerval $[0,100]$. `Synthetic.py` applies $\texttt{PrivFuncSeg}$ with polynomial basis of degree 1, 4, 8, 16. `SyntheticAdapt.py` adopts $`\texttt{PrivFuncSelect}`$.

### CRAWDAD Cab Mobility Dataset

The taxi trajectory dataset is located under `cabspottingdata/`. There are 536 trajectories in the dataset, each with the mobility information of a specific cab during a time period of 20 to 40 days.

`PreprocCabData.py` preprocesses the data by converting latitude and longitude into coordinates (with meter as unit) and removing erroneous datapoints. `SelectCabData.py` truncates the curves by the 12-hour time window "8:00 ~ 20:00, 18 May, 2008", and filters out those with less than 500 samples. There will be 304 curves left, and each curve is considered as a 2D time series.

`TaxiTrajectory.py` privatizes the selected curves with 2D linear basis, and record the $l_2$-losses. Directly calling `TaxiTrajectory.py` will execute in interactive mode, where one example curve of 100 points is privatized under GP with $\varepsilon=0.01$ budget. For batch experiment, add 3 arguments when calling the script: the privacy quota, the batch size, and the SVT threshold factor. Each curve is privatized and recorded for 30 times. The results will be stored under `results/TaxiTrajectory/taxi_{args}/` directory.

```
python PreprocCabData.py
python SelectCabData.py
python TaxiTrajectory.py 0.01 1 > results/taxi_cmd.log
```

### PTB-XL ECG Dataset

The ECG dataset is located under `ptb-xl/`. Since we will only use the frist 100 records of 100 Hz frequency, we only need the records under `ptb-xl/records100/00000/` directory. The records are scaled to the unit of microvolt (μv). Here we use public info of average QRS interval length and scale the time by a factor of 80. Each ECG record consists of 1000 datapoints and spans 10 seconds (thus 800 seconds after scaling).

`ECG.py` loads the records and privatizes them with bounded/unbounded sinc basis. For bounded sinc basis, the time range is split into $m$ intervals, where each interval has bounded sinc basis with shift $800/m$. Directly calling `ECG.py` will execute in interactive mode with $\varepsilon=1$, where unbounded sinc basis is applied by default. For batch experiment, add two arguments when calling the script: the privacy quota and the batch size. When the batch size inputted is $-1$, unbounded sinc basis will be applied; otherwise, corresponding bounded sinc basis is used: given batch size $s\in\mathbb{N}^*$, the corresponding $m$ is $m=1000/s$. Each curve is privatized and recorded for 30 times. The results will be stored under `results/ECG/ECG_{args}/` directory.

```
python ECG.py 1.0 20 > results/ECG_cmd.log
python ECG.py 1.0 -1 > results/ECG_unbounded_cmd.log
```

### Baseline Methods and Other Scripts

The baseline methods for synthetic Gaussian curves are directly included in `Synthetic.py` and `SyntheticAdapt.py`. The number of sample used is $10,20,50,100$, and the smoothing window size is $0.05\times(\text{num of samples})$. Calling those scripts will run our mechanisms along with the baseline methods, and plot figures for comparison.

Baseline methods for CRAWDAD dataset and PTB-XL dataset are implemented in `Baseline.py`. It accepts three arguments: $\texttt{Taxi}$ or $\texttt{ECG}$ to specify the target dataset, a sample rate, and a window scale. The number of sample datapoints drawn is $(\texttt{sample rate})\times(\texttt{total num of datapoints})$, and the window size is $(\texttt{window scale})\times(\texttt{num of samples})$.

After the experiment, use `ExptStats.py` with argument $\texttt{Taxi}$ or $\texttt{ECG}$ to generate summary statistics table for the datasets.

`ExptFigs.py` plots the line graphs of CRAWDAD dataset and PTB-XL dataset. It includes both $l_2$-error and MSE, and the baseline methods' results reported are the best among all choices of sample rate and window scale.

The shell script `expt.sh` runs all experiments with our algorithm and baseline methods, generate result statistics, and plot line graphs for comparisons.