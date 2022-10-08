# ALBADross
ALBADross: Active Learning Based Anomaly Diagnosis for Production HPC Systems

Diagnosing causes of performance variations in High-Performance Computing (HPC) systems is a daunting challenge due to the systems' scale and complexity. Variations in application performance result in premature job termination, lower energy efficiency, or wasted computing resources. One potential solution is manual root-cause analysis based on system telemetry data. However, this approach has become an increasingly time-consuming procedure as the process relies on human expertise and the size of telemetry data is voluminous. Recent research employs supervised machine learning (ML) models to diagnose previously encountered performance anomalies in compute nodes automatically. However, these models generally necessitate vast amounts of labeled samples that represent anomalous and healthy states of an application during training. The demand for labeled samples is constraining because gathering labeled samples is difficult and costly, especially considering anomalies that occur infrequently.

This paper proposes a novel active learning-based framework that diagnoses previously encountered performance anomalies in HPC systems using significantly fewer labeled samples compared to state-of-the-art ML-based frameworks. Our framework combines an active learning-based query strategy and a supervised classifier to minimize the number of labeled samples required to achieve a target performance score. We evaluate our framework on a production HPC system and a testbed HPC cluster using real and proxy applications. We show that our framework, ALBADross, achieves a 0.95 F1-score using 28x fewer labeled samples compared to a supervised approach with equal F1-score, even when there are previously unseen applications and application inputs in the test dataset.

Maintainer: 
* **Burak Aksar** - *baksar@bu.edu* 

Developers:  
* **Burak Aksar** - *baksar@bu.edu* & **Efe Sencan** - *esencan@bu.edu* 


## Installation

Install the requirements. The most stable Python version is 3.6.5. However, it works with 3.7x and 3.8x as well. Please note that we didn't test the code with Python 3.9x.


1-) Create a local virtual environment in the folder

```
python3 -m venv ml_venv
```

2-) Activate venv

```
source ml_venv/bin/activate/
```

3-) Install requirements

```
pip3 install --user -r albadross_reqs.txt
```

If you want to use TSFRESH and MVTS feature extraction methods, please refer to following links: 

* [TSFRESH Repo for Installation Instructions](https://github.com/blue-yonder/tsfresh)

* [MVTS Toolkit](https://github.com/ElsevierSoftwareX/SOFTX_2020_15)



## Authors

CLUSTER'22: [ALBADross: Active Learning Based Anomaly Diagnosis for Production HPC Systems](https://www.bu.edu/peaclab/files/2022/10/ALBADross_Cluster_22_CR.pdf)

Authors:
    Burak Aksar (1), Efe Sencan(1), Benjamin Schwaller (2),  Omar Aaziz (2), Vitus J. Leung (2), Jim Brandt (2), Brian Kulis (1), Ayse K. Coskun (1)

Affiliations:
    (1) Department of Electrical and Computer Engineering, Boston University
    (2) Sandia National Laboratories

This work has been partially funded by Sandia National Laboratories. Sandia
National Laboratories is a multimission laboratory managed and operated by
National Technology and Engineering Solutions of Sandia, LLC., a wholly owned
subsidiary of Honeywell International, Inc., for the U.S. Department of
Energy’s National Nuclear Security Administration under Contract DENA0003525.


## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details
