import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
loss_resnet = pd.read_csv("records/q2/resnet_80/resnet_best_loss.csv")


steps = loss_resnet["Step"]

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(steps, loss_resnet["Value"])
plt.title("ResNet Loss Curves")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.savefig("./visualizations/loss_curves_resnet.png")

map_resnet = pd.read_csv("records/q2/resnet_80/resnet_best_map.csv")

steps = map_resnet["Step"]
best_map = np.max(map_resnet["Value"])
best_step = steps[np.argmax(map_resnet["Value"])]
# Plot mAP curve
plt.figure(figsize=(10, 6))
plt.plot(steps, map_resnet["Value"])
plt.scatter(best_step, best_map, color="red")

plt.text(
    best_step,
    best_map,
    f"{best_map:.2f}",
    fontsize=12,
    verticalalignment="bottom",
    horizontalalignment="right",
)
plt.title("ResNet mAP Curves")
plt.xlabel("Steps")
plt.ylabel("mAP")
plt.savefig("./visualizations/map_curves_resnet.png")

lr_resnet = pd.read_csv("records/q2/resnet_80/resnet_best_lr.csv")
steps = lr_resnet["Step"]

# Plot learning rate curve
plt.figure(figsize=(10, 6))
plt.plot(steps, lr_resnet["Value"])
plt.title("ResNet Learning Rate Curve")
plt.xlabel("Steps")
plt.ylabel("Learning Rate")
plt.savefig("./visualizations/lr_curve_resnet.png")
