import argparse
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def json_to_parquet(src, dst):
    with open(src, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # drill into the array
    data = raw.get("colors", [])

    df = pd.json_normalize(data)[["name", "hex", "bestContrast"]]

    table = pa.Table.from_pandas(df)
    pq.write_table(table, dst)

    print("JSON rows:", len(data))
    print("Parquet rows:", df.shape[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json_path")
    p.add_argument("parquet_path")
    args = p.parse_args()

    json_to_parquet(args.json_path, args.parquet_path)
    print("done")


if __name__ == "__main__":
    main()
