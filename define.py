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
    "AIT_400_001A_P.PV": "intake conductivity (uS/cm)", 
    "FIT_400_001A_P.PV": "wastewater flowrate (GPM)",
    "FIT_400_002A_P.PV": "intake flowrate (GPM)",
    "P_400_002A_P.RPM_MV": "HP Pump speed (RPM)",
    "PT_400_002A_P.PV": "HP Pump pressure (PSI)",
    "P_400_001A_P.RPM_MV": "Circulation Pump speed (RPM)",
    "PDIT_400_002A_P.PV": "Circulation Pump pressure (PSI)",
}

RO_name_to_color = {
    "intake conductivity (uS/cm)": "#1f78b4",
    "wastewater flowrate (GPM)": "#33a02c",
    "intake flowrate (GPM)": "#1f78b4",
    "HP Pump speed (RPM)": "#1f78b4",
    "HP Pump pressure (PSI)": "#1f78b4",
    "Circulation Pump speed (RPM)": "#33a02c",
    "Circulation Pump pressure (PSI)": "#33a02c",
}

RO_item_to_color = {
    "intake": "#1f78b4",
    "wastewater": "#33a02c",
    "HP Pump": "#1f78b4",
    "Circulation Pump": "#33a02c",
}