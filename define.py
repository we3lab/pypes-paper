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
    "intake conductivity (uS/cm)": "red",
    "wastewater flowrate (GPM)": "saddlebrown",
    "intake flowrate (GPM)": "saddlebrown",
    "HP Pump speed (RPM)": "green",
    "HP Pump pressure (PSI)": "black",
    "Circulation Pump speed (RPM)": "green",
    "Circulation Pump pressure (PSI)": "black",
}

RO_item_to_color = {
    "intake": "blue",
    "wastewater": "orange",
    "HP Pump": "saddlebrown",
    "Circulation Pump": "green",
}