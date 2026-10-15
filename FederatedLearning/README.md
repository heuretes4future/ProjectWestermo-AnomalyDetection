Preprocessing for datasets

The main preprocessing code runs by shell with 2 arguments: (dataset.csv, config.py). Config being optional otherwise ran on default preprocessing mode and returns a preprocessed file in the same folder as shell directory. 
For specific preprocessing change variables inside the config\_file, this requires that you know the exact names to be removed. 
For the script to work properly the datasets need to be inside the same folder as the script and config file.
The script is configured with multiple classes and can be expanded in the future, but the config file must reflect those changes.



Running Federated Learning

This project uses the Flower framework to run federated learning in a Linux-based environment. All testing has been performed on WSL2 (Windows Subsystem for Linux) on Windows.

To start the federated learning system, run the following command in your terminal:

docker compose up --build

Each client expects its own preprocessed dataset stored as a CSV file. The current setup uses three clients:

- client-left

- client-right

- client-bottom

These datasets must be generated before running federated learning.

Federated learning currently uses a placeholder Random Forest model.