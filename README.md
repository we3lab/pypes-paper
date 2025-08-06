# pypes-paper
Case studies and software applications for the PyPES journal paper

## Optimal Sensor Placement
In this case study, we implemented the algorithm from the paper ["Optimal flow sensor placement on wastewater treatment plants"](https://doi.org/10.1016/j.watres.2016.05.068) by Villez et al. to demonstrate how PyPES can be used to implement graph-based algorithms in a facility-agnostic fashion. The wastewater model is derived from [Benchmark Simulation Model no. 2 (BSM2)](https://doi.org/10.2166/9781780401171), which was originally developed by Gernaey et al.

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
In this case study, we implemented the algorithm from the paper ["Advanced monitoring of water systems using in situ measurement stations: data validation and fault detection"](https://doi.org/10.2166/wst.2013.302) by Alferes et al. to demonstrate how PyPES can ease the implementation of complex fault detection algorithms by managing data and metadata. The reverse osmosis treatment train comes from a real world desalination plant.

### Usage
```
python fault_detection.py [arguments]
```

#### Arguments
- `--tag_file` or `-t`: Specifies the path to the tag file with information about the sensors data.
- `--data_folder` or `-d`: Specifies the path to the data folder. Inside this folder, there should be one or more CSV files containing the data to be analyzed.
- `--train_test_split` or `-s`: Specifies the train-test split ratio. Default is 0.8.
- `--data_plot` or `-p`: Specifies the path to save the data plot. Default is `results/fault_detection/data.png`.
- `--json_path` or `-j`: Specifies the path to the network JSON file. Default is `json/desalination.json`.
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
In this case study, we implemented the algorithm from the paper ["Leakage fault detection in district metered areas of water distribution systems "](https://doi.org/10.2166/hydro.2012.109) by Eliades & Polycarpou to demonstrate how a user can automatically convert EPANet files into a PyPES model and then conduct analysis on a large number of nodes and connections. 

The distribution system model comes from [Dataset of BattLeDIM: Battle of the Leakage Detection and Isolation Methods](https://doi.org/10.5281/zenodo.4017658). The original distribution network, `L-TOWN.inp` and the converted PyPES model (`distribution.json`) are included in the `json` folder under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode) license, including the [Disclaimer of Warranties and Limitation of Liability](https://creativecommons.org/licenses/by/4.0/legalcode#s5). The citation, including full list of authors, can be found in [References](#references).

### Usage
To convert an EPANet file:
```
python epyt_utils.py json/L-TOWN.inp json/distribution.json
```

To run the leakage detection algorithm:
```
python leakage_detection.py
```

## References

&nbsp; Alferes, J., Tik, S., Copp, J. and Vanrolleghem, P.A., 2013. Advanced monitoring of water systems using in situ measurement stations: data validation and fault detection. *Water Science and Technology*, 68(5), pp.1022-1030. https://doi.org/10.2166/wst.2013.302 

&nbsp; Eliades, D.G. and Polycarpou, M.M., 2012. Leakage fault detection in district metered areas of water distribution systems. *Journal of Hydroinformatics*, 14(4), pp.992-1005. https://doi.org/10.2166/hydro.2012.109

&nbsp; Gernaey, K.V., Jeppsson, U., Vanrolleghem, P.A., Copp, J.B. (Eds.), 2014. *Benchmarking of Control Strategies for Wastewater Treatment Plants*. IWA Publishing. https://doi.org/10.2166/9781780401171

&nbsp; Villez, K., Vanrolleghem, P.A., Corominas, L., 2016. Optimal flow sensor placement on wastewater treatment plants. *Water Research*, 101, pp.75–83. https://doi.org/10.1016/j.watres.2016.05.068

&nbsp; Vrachimis, S.G., Eliades, D.G., Taormina, R., Ostfeld, A., Kapelan, Z., Liu, S., Kyriakou, M.S., Pavlou, P., Qiu, M., Polycarpou, M., 2020. *Dataset of BattLeDIM: Battle of the Leakage Detection and Isolation Methods*. https://doi.org/10.5281/ZENODO.4017658

## Acknowledgments

This work is supported by the National Alliance for Water Innovation (NAWI), funded by the U.S. Department of Energy, Energy Efficiency and Renewable Energy Office, Advanced Manufacturing Office under Funding Opportunity Announcement DE-FOA-0001905. The views expressed herein do not necessarily represent the views of the U.S. Department of Energy or the United States Government. 

This work is also supported by the Center for Integrated Facility Engineering at Stanford University as a part of CIFE Seed Proposal 2023-02. 

We would like to thank the staff of the Charles Meyer Desalination Plant for reverse osmosis data and metadata and our collaborator Akshay Rao for insight into the reverse osmosis system.