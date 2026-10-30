import pandas as pd

for filename in ["data/q1/ecg_train.csv", "data/q1/ecg_val.csv"]:
    df = pd.read_csv(filename)

    print("\n" + "=" * 65)
    print(filename)
    print("=" * 65)

    print("\nw5_mean statistics by label:")
    print(df.groupby("label")["w5_mean"].agg(
        ["count", "min", "max", "mean", "std"]
    ).to_string())

    print("\nLabel counts:")
    print(df["label"].value_counts().sort_index().to_string())

    print("\nFirst 10 rows of w5_mean and label:")
    print(df[["w5_mean", "label"]].head(10).to_string(index=False))
