import numpy as np
import pandas as pd


def pivotTable(weather: pd.DataFrame) -> pd.DataFrame:
    return weather.pivot(index='month', columns='city', values='temperature')


if __name__ == "__main__":
    weather: pd.DataFrame = pd.read_csv('pandas dataset/2889. Reshape Data: Pivot.csv')
    print(pivotTable(weather))
