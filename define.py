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
    "Permeate conductivity (uS/cm)": "#A6CEE4",
    "Brine flowrate (GPM)": "#B2E08A",
    "Permeate flowrate (GPM)": "#A6CEE4",
    "HP pump speed (Hz)": "#A6CEE4",
    "HP pump pressure (PSI)": "#A6CEE4",
    "Circulation pump speed (Hz)": "#B2E08A",
    "Circulation pump pressure (PSI)": "#B2E08A",
}

RO_item_to_color = {
    "Permeate": "#A6CEE4",
    "Brine": "#B2E08A",
    "HP pump": "#A6CEE4",
    "Circulation pump": "#B2E08A",
}