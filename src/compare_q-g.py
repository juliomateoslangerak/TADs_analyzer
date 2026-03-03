import pandas as pd

quentin_df = pd.read_csv('/home/julio/Documents/data-annotation/Image-Data-Annotation/scripts/quentin_measurement_file.csv')
georges_df = pd.read_csv('/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/CTCF-AID/analysis_df.csv')

georges_df = georges_df.drop(georges_df[georges_df['Channel ID'] == 0].index)
georges_df = georges_df.drop(georges_df[georges_df['roi_type'] == 'subdomain'].index)
georges_df = georges_df.drop(georges_df[georges_df['roi_type'] == 'overlap'].index)

georges_df['Image Name'] = georges_df['Image Name'].apply(lambda x: x[:-9])

merged_df = pd.merge(quentin_df, georges_df, left_on='on', right_on='Image Name')
merged_df["Volume_diff_factor"] = merged_df['A565_volume'] / merged_df['volume']
merged_df["Principal_axis_diff"] = merged_df['A565_principal_axis'] - merged_df['volume']
merged_df["Distance_diff"] = merged_df['distance'] - merged_df['volume']
merged_df["Volume_diff"] = merged_df['A565_volume'] - merged_df['volume']

pass
