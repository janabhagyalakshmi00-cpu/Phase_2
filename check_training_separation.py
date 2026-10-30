import pandas as pd

for filename in ["data/q1/ecg_train.csv", "data/q1/ecg_val.csv"]:
    df = pd.read_csv(filename)
    print(f"\n{filename}")

    stats = df.groupby("label")["w5_mean"].agg(["count", "min", "max", "mean"])
    print(stats.to_string())

    label_0_max = df.loc[df["label"] == 0, "w5_mean"].max()
    label_1_min = df.loc[df["label"] == 1, "w5_mean"].min()
    label_1_max = df.loc[df["label"] == 1, "w5_mean"].max()
    label_0_min = df.loc[df["label"] == 0, "w5_mean"].min()

    print(f"Label 0 minimum: {label_0_min:.6f}")
    print(f"Label 0 maximum: {label_0_max:.6f}")
    print(f"Label 1 minimum: {label_1_min:.6f}")
    print(f"Label 1 maximum: {label_1_max:.6f}")
