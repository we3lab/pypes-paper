# pypes-paper
Case studies and software applications for the PyPES journal paper

## Optimal Sensor Placement
### Usage
```
python optimal_sensor_placement.py [arguments]
```
#### Arguments
- `--mode` or `-m`: Specifies which Waste Water Treatment Plant (WWTP) to use. It accepts an integer value (1, 2, or 3) and defaults to 1 if not provided.
- `--update` or `-u`: A flag that, when set, enables update mode.
- `--log_file` or `-l`: Specifies the path to the log file where the results will be saved.
- `--original_log_file` or `-o`: Specifies the path to the original log file for comparison purposes in update mode.:

### Example
To run the script with the default WWTP (1) and save results to `results.txt`:
```bash
python optimal_sensor_placement.py --log_file results.txt
```
To run the script with WWTP 1 with update mode enabled and save results to `results_update.txt`:
```bash
python optimal_sensor_placement.py --mode 1 --update --log_file results_update.txt --original_log_file results.txt
```

## Fault Detection
### Usage
```
python fault_detection.py [arguments]
```

#### Arguments
- `--tag_file` or `-t`: Specifies the path to the tag file with information about the sensors data.
- `--data_folder` or `-d`: Specifies the path to the data folder. Inside this folder, there should be one or more CSV files containing the data to be analyzed.
- `--train_test_split` or `-s`: Specifies the train-test split ratio. Default is 0.8.
- `--data_plot` or `-p`: Specifies the path to save the data plot. Default is `results/fault_detection/data.png`.
- `--json_path` or `-j`: Specifies the path to the network JSON file. Default is `json/Desal.json`.
- `--T2Q_plot` or `-t`: Specifies the path to save the T2Q plot. Default is `results/fault_detection/T2Q_plot.png`.
- `--PC_plot` or `-c`: Specifies the path to save the PC plot. Default is `results/fault_detection/PC_plot.png`.
### Example
To run the script with the default parameters:
```bash
python fault_detection.py
```
To run the script with a custom tag file and data folder:
```bash
python fault_detection.py --tag_file custom_tag.csv --data_folder custom_data
```

## Leakage Detection
### Usage
```
python leakage_detection.py
```


