from sklearn.preprocessing import MinMaxScaler


def group_data(Data):
 grouped_data = Data.groupby('trackId')
 separated_data = {}

 for track_id, group in grouped_data:
    # Extract relevant parameters (e.g., longitudinal and lateral velocity)
     longitudinal_velocity = group['lonVelocity'].values
     lateral_velocity = group['latVelocity'].values
     xCenter = group['xCenter'].values
     yCenter = group['xCenter'].values
     heading = group['heading'].values
     width = group['width'].values
     length = group['length'].values
     xVelocity = group['xVelocity'].values
     yVelocity = group['yVelocity'].values
     xAcceleration = group['xAcceleration'].values
     yAcceleration = group['yAcceleration'].values
     lonAcceleration = group['lonAcceleration'].values
     latAcceleration = group['latAcceleration'].values

     # Normalize the parameters
     scaler = MinMaxScaler(feature_range=(-1, 1))
     longitudinal_velocity_normalized = scaler.fit_transform(
         longitudinal_velocity.reshape(-1, 1)).flatten()
     lateral_velocity_normalized = scaler.fit_transform(
         lateral_velocity.reshape(-1, 1)).flatten()
     xCenter_normalized = scaler.fit_transform(
         xCenter.reshape(-1, 1)).flatten()
     yCenter_normalized = scaler.fit_transform(
         yCenter.reshape(-1, 1)).flatten()
     heading_normalized = scaler.fit_transform(
         heading.reshape(-1, 1)).flatten()
     width_normalized = scaler.fit_transform(
         width.reshape(-1, 1)).flatten()
     length_normalized = scaler.fit_transform(
         length.reshape(-1, 1)).flatten()
     xVelocity_normalized = scaler.fit_transform(
         xVelocity.reshape(-1, 1)).flatten()
     yVelocity_normalized = scaler.fit_transform(
         yVelocity.reshape(-1, 1)).flatten()
     xAcceleration_normalized = scaler.fit_transform(
         xAcceleration.reshape(-1, 1)).flatten()
     yAcceleration_normalized = scaler.fit_transform(
         yAcceleration.reshape(-1, 1)).flatten()
     lonAcceleration_normalized = scaler.fit_transform(
         lonAcceleration.reshape(-1, 1)).flatten()
     latAcceleration_normalized = scaler.fit_transform(
         latAcceleration.reshape(-1, 1)).flatten()

     # Store the normalized parameters in the dictionary
     separated_data[track_id] = {
         'lonVelocity': longitudinal_velocity_normalized,
         'latVelocity': lateral_velocity_normalized,
         'xCenter': xCenter_normalized,
         'yCenter': yCenter_normalized,
         'heading': heading_normalized,
         'width': width_normalized,
         'length': length_normalized,
         'xVelocity': xVelocity_normalized,
         'yVelocity': yVelocity_normalized,
         'xAcceleration': xAcceleration_normalized,
         'yAcceleration': yAcceleration_normalized,
         'lonAcceleration': lonAcceleration_normalized,
         'latAcceleration': latAcceleration_normalized,
         # Add other parameters as needed
     }
 return separated_data
