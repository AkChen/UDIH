Unsupervised Deep Imputed Hashing(UDIH) for partial cross-modal retrieval - Tensorflow
===
Prerequisites
---
#####. Linux or Windows

#####. Python 3

#####. Tensorflow

#####. Numpy

Getting started
---
####. Clone this repo:
```Java
git clone https://github.com/AkChen/UDIH
cd UDIH
```
####. Downlowd the dataset (MIRFlickr & NUS-WIDE)
```Java
https://pan.baidu.com/s/1A9ZLU8l-PKJ0xN8kLWkAFQ  pwd:j5gs 
```

Put the .mat file under _<UDIH_DIR>/data/_

####. Generate Partial Data
```Java
cd PCMH-mir
python  generate_partial_data_mir.py
```

Once the partial data is generated, we just need focus on Imputation and Hashing learning.
####. Imputation

Set P_i (default: 0.02) and P_t (default: 0.01) _in PCMH_2Path_Imputation_MIR.py_
```Java
python PCMH_2Path_Imputation_MIR.py
```
####. Hashing Learning

Set the length of hash code (default: 16) _BIT_
```Java
python PCMH_Hashing_Leaning_MIR_16.py
```
The results will be recorded in _<UDIH_DIR>/record_

