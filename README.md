# Signal detection

## Basics
Designed for Cavity-Ring-Down-Spectrum(crds) detection. The main function in *probe.py* calculate the distance similarity with the bunch of known patterns.

Expected *.txt data column format: <time>, <signal crds voltage>, <signal triangle-wave voltage>
or
expected *.npy data shape is (3, N) for <time>, <signal crds voltage>, <signal triangle-wave voltage>

## Requirements (Python3.8+)
>. pip3 <or python3 -m pip> install -r requirements.txt

## How to use
> python3 probe.py --config=./config/probe.json
    >  You need to change the folder config/probe.json raw_data: *.txt

## TODO list
- [] Add evaluation 
- [] Fast fitting exponential script