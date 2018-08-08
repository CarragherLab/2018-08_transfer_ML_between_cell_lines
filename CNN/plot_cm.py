import sys
import json
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(["seaborn-paper", "seaborn-ticks"])

with open(sys.argv[1]) as f:
    data_dict = json.load(f)


for cell_line, data in data_dict.items():
    data_arr = np.asarray(data["cm"])
    labels = [i.replace("_", " ") for i in data["labels"]]
    plt.figure()
    plt.grid(linestyle=":")
    plt.imshow(data_arr, cmap=plt.cm.bone_r, vmin=0, vmax=1)
    plt.colorbar()
    plt.xticks(range(data_arr.shape[0]), labels, rotation=90)
    plt.yticks(range(data_arr.shape[1]), labels)
    plt.title("{}\naccuracy = {:.2f}%".format(cell_line, data["acc"]*100),
               loc="left", fontweight="bold")
    plt.tight_layout()
    name = "confusion_matrix_{}.pdf".format(cell_line)
    plt.savefig(name)
