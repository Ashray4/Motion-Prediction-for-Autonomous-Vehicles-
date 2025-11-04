import pandas as pd

def read_dataset(file_path):
    return pd.read_csv(file_path)




# track_id_example = list(separated_data.keys())[0]
# print(f"Track ID: {track_id_example}")
# print("Longitudinal Velocity:",
#       separated_data[track_id_example]['longitudinal_velocity'])
# print("Lateral Velocity:",
#       separated_data[track_id_example]['lateral_velocity'])

# with open('data/processed/', 'wb') as file:
#     pickle.dump(Data_DS, file)
