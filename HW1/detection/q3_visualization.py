import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

box_loss = pd.read_csv("../records/q3/overfit/box_loss.csv", index_col=0)
center_loss = pd.read_csv("../records/q3/overfit/center_loss.csv", index_col=0)
class_loss = pd.read_csv("../records/q3/overfit/class_loss.csv", index_col=0)

total_loss = (
    np.array(list(class_loss["Value"]))
    + np.array(list(center_loss["Value"]))
    + np.array(list(box_loss["Value"]))
)
plt.figure(figsize=(10, 6))
plt.plot(box_loss["Step"], box_loss["Value"], label="Box Loss")
plt.plot(center_loss["Step"], center_loss["Value"], label="Center Loss")
plt.plot(class_loss["Step"], class_loss["Value"], label="Class Loss")
plt.plot(
    class_loss["Step"],
    total_loss,
    label="Total Loss",
)

plt.title("Loss Curves")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.savefig("../visualizations/q3_loss_curves_overfit.png")

box_loss = pd.read_csv("../records/q3/full2/box_loss.csv", index_col=0)
center_loss = pd.read_csv("../records/q3/full2/ctr_loss.csv", index_col=0)
class_loss = pd.read_csv("../records/q3/full2/class_loss.csv", index_col=0)

total_loss = (
    np.array(list(class_loss["Value"]))
    + np.array(list(center_loss["Value"]))
    + np.array(list(box_loss["Value"]))
)
plt.figure(figsize=(10, 6))
plt.plot(box_loss["Step"], box_loss["Value"], label="Box Loss")
plt.plot(center_loss["Step"], center_loss["Value"], label="Center Loss")
plt.plot(class_loss["Step"], class_loss["Value"], label="Class Loss")
plt.plot(
    class_loss["Step"],
    total_loss,
    label="Total Loss",
)

plt.title("Loss Curves")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.savefig("../visualizations/q3_loss_curves_full.png")
