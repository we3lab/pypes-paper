color_map = {
    "Electricity": "yellow",
    "UntreatedSewage": "saddlebrown",
    "PrimaryEffluent": "saddlebrown",
    "SecondaryEffluent": "saddlebrown",
    "TertiaryEffluent": "saddlebrown",
    "TreatedSewage": "green",
    "WasteActivatedSludge": "black",
    "PrimarySludge": "black",
    "TWAS": "black",
    "TPS": "black",
    "Scum": "black",
    "SludgeBlend": "black",
    "ThickenedSludgeBlend": "black",
    "Biogas": "red",
    "GasBlend": "red",
    "NaturalGas": "gray",
    "Seawater": "aqua",
    "Brine": "aqua",
    "SurfaceWater": "cornflowerblue",
    "Groundwater": "cornflowerblue",
    "Stormwater": "cornflowerblue",
    "NonpotableReuse": "purple",
    "DrinkingWater": "blue",
    "PotableReuse": "blue",
    "FatOilGrease": "orange",
    "FoodWaste": "orange",
}

RO_tag_mappping = {
    "From date": "timestamp",
    "AIT_400_001A_P.PV": "Permeate conductivity (uS/cm)", 
    "FIT_400_001A_P.PV": "Brine flowrate (GPM)",
    "FIT_400_002A_P.PV": "Permeate flowrate (GPM)",
    "P_400_002A_P.RPM_MV": "HP pump speed (Hz)",
    "PIT_400_003A_P.PV": "HP pump pressure (PSI)",
    "P_400_001A_P.RPM_MV": "Circulation pump speed (Hz)",
    "PDIT_400_002A_P.PV": "Circulation pump pressure (PSI)",
}

RO_name_to_color = {
    "Permeate conductivity (uS/cm)": "#61CBF4",
    "Brine flowrate (GPM)": "#92D050",
    "Permeate flowrate (GPM)": "#61CBF4",
    "HP pump speed (Hz)": "#61CBF4",
    "HP pump pressure (PSI)": "#61CBF4",
    "Circulation pump speed (Hz)": "#92D050",
    "Circulation pump pressure (PSI)": "#92D050",
}

RO_item_to_color = {
    "Permeate": "#61CBF4",
    "Brine": "#92D050",
    "HP pump": "#61CBF4",
    "Circulation pump": "#92D050",
}