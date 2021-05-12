from file_handling_docker import *
from functions import *
from math import log
import pandas as pd
from sklearn import metrics


print("Program: Phase Prediction")
print("Release: 1.0.0")
print("Date: 2021-05-12")
print("Author: Brian Neely")
print()
print()
print("This program reads a csv file and will preform mutual information calculation on the combination of meters and ")
print("transformer phases.")
print()
print()

# Open CSV
data = open_unknown_csv("data.csv", ',')

# Get column names of meters, circuit phases, date_time, meter voltage, and phase voltage
meter_col = "meter_id"
phase_transformer_col = "phase_transformer_id"
date_time_col = "datetime"
meter_volt_col = "meter_voltage"
phase_volt_col = "phase_transformer_voltage"

# Assign mi column
mi_col = "mi"

# Get list of meters/phases
meter_phase_list = data[[meter_col, phase_transformer_col]].drop_duplicates()

# Reset index on meter_phase_list
meter_phase_list = meter_phase_list.reset_index(drop=True)

# Make a database of results
data_calcd = pd.DataFrame(columns=[meter_col, phase_transformer_col, mi_col])

# Loop through meter/phase combinations
for index, row in meter_phase_list.iterrows():
	# Set meter and phases
	meter_mi = row[meter_col]
	phase_mi = row[phase_transformer_col]

	# Get dataframe for combination
	data_mi = data[(data[meter_col] == meter_mi) & (data[phase_transformer_col] == phase_mi)]

	# Calculate mi for combination
	mi_calc = metrics.mutual_info_score(data_mi[meter_volt_col].tolist(), data_mi[phase_volt_col].tolist())

	# Make row
	out_row = pd.DataFrame([[meter_mi, phase_mi, mi_calc]], columns=[meter_col, phase_transformer_col, mi_col])

	# Append row
	data_calcd = data_calcd.append(out_row)

# Write Results
data_calcd.to_csv("data_out.csv")
