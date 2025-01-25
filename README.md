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
- `--original_log_file` or `-o`: Specifies the path to the original log file for comparison purposes in update mode.

### Example
```
python optimal_sensor_placement.py --mode 2 --log_file results.txt
```