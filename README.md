# ProjectWestermo

### Datasets

**1. The Westermo Dataset**
This dataset consists of network traffic data from three different network nodes.
* **Download source:** [Westermo Network Traffic Dataset (GitHub)](https://github.com/westermo/network-traffic-dataset)
* **Note:** Make sure to download the **reduced** dataset files (`output_right.csv`, `output_bottom.csv` and `output_left.csv`).

**2. The Edge-IIoT Dataset**
A comprehensive cyber security dataset specifically designed for IoT and IIoT applications.
* **Download source:** [Edge-IIoTset on Kaggle](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot?resource=download)
* **Note:** Extract and use the file named `ML-EdgeIIoT-dataset.csv`.


The execution order depends on which dataset you are processing:

**For the Westermo dataset:**
1. Run `preprocessing.py` to prepare the data for training.

**For the Edge-IIoT dataset:**
1. Run the partition script first to split and organize the data correctly.
2. Run `preprocessing.py` to prepare the data for training.
