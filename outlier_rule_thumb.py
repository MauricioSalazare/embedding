import pandas as pd

# %%

dali_data = pd.read_csv("data/processed/rlps_2023_data.csv")


#%%
dali = dali_data.drop(columns=["GEMEENTE", "CLUSTER", "BOXID", "MONTH"], errors="ignore")
idx = (dali < 15).all(axis=1)

dali_clean = dali_data[~idx]
counts = dali_clean[dali_clean["MONTH"]==1]["GEMEENTE"].value_counts()

print(counts)
print(f"Total: {counts.sum()}")

selected_gemeentes = counts[:10].index.tolist()
idx_cities = dali_clean["GEMEENTE"].isin(selected_gemeentes)
dali_double_clean = dali_clean[idx_cities]
dali_double_clean.to_csv("data/processed/rlps_2023_data_clean.csv", index=False)


#%%
# # %% =====================================================
# # RULE OF THUMB TO DROP OUTLIERS
# YEAR_COUNTS = 365 * 24 * 4  # 15-minute resolution
# PER_COMPLETE = 0.95
# MIN_POWER = 30  # kW - Minimum maximum power required to be a valid measurement
#
# dali_boxes_ids = dali_data["BOXID"].unique().tolist()
#
# # Gemeente-BOXID, unique combinations
# unique_combinations = dali_data[["GEMEENTE", "BOXID"]].drop_duplicates()
# unique_combinations = unique_combinations.reset_index(drop=True)
#
# # Insufficient readings
# readings_counts = dali_data["BOXID"].value_counts(dropna=True)
# dali_complete = readings_counts[readings_counts >= YEAR_COUNTS * PER_COMPLETE].copy()
#
# # Readings very low (DALI boxes that in the whole year at leas has a peak on 30 kW)
# reading_maximum = dali_data[["POWER", "BOXID"]].groupby("BOXID", dropna=True).max()
# dali_high_power_values = reading_maximum[reading_maximum > MIN_POWER].dropna()
#
# dali_correct = set(dali_complete.index).intersection(set(dali_high_power_values.index))
# dali_incorrect = set(dali_boxes_ids).difference(dali_correct)
#
# dali_incorrect_frame = pd.DataFrame(dali_incorrect, columns=["BOXID-wrong"])
# dali_incorrect_frame.to_csv("data/processed/dali_incorrect_2023.csv", index=False, header=False)
#
# # Perform an inner join between the two DataFrames
# result_gemeente = pd.merge(
#     dali_incorrect_frame.rename(columns={"BOXID-wrong": "BOXID"}),  # Rename column for join
#     unique_combinations,
#     on="BOXID",
#     how="inner",
# )
#
# # Reset index for cleaner output (optional)
# result_gemeente = result_gemeente.reset_index(drop=True)
# result_gemeente = result_gemeente.sort_values(by="GEMEENTE", ascending=True).reset_index(drop=True)
# result_gemeente.rename(columns={"BOXID": "BOXID-wrong"}, inplace=True)
# result_gemeente.to_csv("data/processed/dali_incorrect_gemeente_2023.csv", index=False, header=False)
#
# # %%
# incorrect_dali = pd.read_csv("data/processed/dali_incorrect_2023.csv", header=None)
#
# # %% Overview of number of DALIBOXES
#
#
# filtered_data = dali_data[dali_data["BOXID"].isin(dali_incorrect)]
# incorrect_counts = filtered_data.groupby("GEMEENTE")["BOXID"].nunique()
#
# total_counts = dali_data.groupby("GEMEENTE")["BOXID"].nunique()
# result = pd.DataFrame(
#     {
#         "Correct_BOXID_Count": total_counts - incorrect_counts,
#         "Incorrect_BOXID_Count": incorrect_counts,
#         "Total_BOXID_Count": total_counts,
#     }
# ).reset_index()
# result["Incorrect_Percentage"] = result["Incorrect_BOXID_Count"] / result["Total_BOXID_Count"] * 100
#
# # %%
# print(result)
