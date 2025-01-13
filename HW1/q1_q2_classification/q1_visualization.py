import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
loss_with_data_aug = pd.read_csv("records/q1/data_aug_map026/q1_with_data_aug_loss.csv")
loss_without_data_aug = pd.read_csv(
    "records/q1/data_no_aug_map024/q1_no_data_aug_loss.csv"
)

steps = loss_with_data_aug["Step"]

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(steps, loss_with_data_aug["Value"], label="With Data Augmentation")
plt.plot(steps, loss_without_data_aug["Value"], label="Without Data Augmentation")
plt.title("Loss Curves")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.savefig("./visualizations/loss_curves.png")

map_with_data_aug = pd.read_csv("records/q1/data_aug_map026/q1_with_data_aug_map.csv")
map_without_data_aug = pd.read_csv(
    "records/q1/data_no_aug_map024/q1_no_data_aug_map.csv"
)
steps = map_with_data_aug["Step"]
# Plot mAP curve
plt.figure(figsize=(10, 6))
plt.plot(steps, map_with_data_aug["Value"], label="With Data Augmentation")
plt.plot(steps, map_without_data_aug["Value"], label="Without Data Augmentation")
plt.title("mAP Curves")
plt.xlabel("Steps")
plt.ylabel("mAP")
plt.legend()
plt.savefig("./visualizations/map_curves.png")

lr_with_data_aug = pd.read_csv("records/q1/data_aug_map026/q1_with_data_aug_lr.csv")
steps = lr_with_data_aug["Step"]

# Plot learning rate curve
plt.figure(figsize=(10, 6))
plt.plot(steps, lr_with_data_aug["Value"])
plt.title("Learning Rate Curve")
plt.xlabel("Steps")
plt.ylabel("Learning Rate")
plt.savefig("./visualizations/lr_curve.png")
