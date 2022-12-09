# Signal detection

## Basics
Designed for Cavity-Ring-Down-Spectrum(crds) detection. The main function in *probe.py* calculate the distance similarity with the bunch of known patterns. The patterns are selected manually.

## Input description
Expected raw data *.txt as input data:
 data column format: time, signal crds voltage, signal triangle-wave voltage
or
Expected the processed *.npy as input data 
 data shape is (3, N) for time, signal crds voltage, signal triangle-wave voltage

## Ouput description
If you run probe.py, it will output and save three *.npy files, tri_peak_idxs.npy, det_idxs.npy and i_idxs.npy.

* tri_peak_idxs.npy : list of np arrays of absolute index for triangle high and low location ex. np.array([100, 200], [300, 400], ..., [n, m])
* det_idxs.npy : list of absolute index in raw data for crds peak location.
* i_idxs.npy : ith window find the crds peak signal

## Requirements (Python3.8+)
> pip3 <or python3 -m pip> install -r requirements.txt

## How to use
> python3 probe.py --config=./config/probe.json
    (You need to change the folder config/probe.json raw_data: *.txt)

## TODO list
- [] Add evaluation 
- [] Fast exponential fitting  script