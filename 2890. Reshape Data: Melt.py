import pandas as pd


def meltTable(report: pd.DataFrame) -> pd.DataFrame:
    # return pd.melt(report, id_vars=['product'], var_name='quarter', value_name='sales')
    data = {
        "product": [],
        "quarter": [],
        "sales": []
    }
    product = report["product"]
    for i in report.columns[1:]:
        for p in product:
            data["product"].append(p)
            data["quarter"].append(i)
            data["sales"].append(report[report["product"] == p][i].values[0])

    return pd.DataFrame(data)


if __name__ == "__main__":
    report = pd.read_csv("pandas dataset/2890. Reshape Data: Melt.csv")
    print(meltTable(report))

# Done ✅
