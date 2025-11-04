# Motion Prediction for Autonomous Vehicles

Learning-based motion prediction for autonomous vehicles using sequence models (LSTM) and a feed-forward baseline (MLP).
The project provides a full pipeline: data loading → preprocessing → training → evaluation → visualization, with reproducible configs.

In the experiments, the LSTM achieved ~30.5% lower trajectory prediction error than the baseline MLP.

Key Features

- Trajectory prediction from past motion histories (positions/velocities)

- Models: LSTM  and MLP baseline

- Visualizations: predicted vs. ground-truth trajectories, error heatmaps
