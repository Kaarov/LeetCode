import pandas as pd


def big_countries(world: pd.DataFrame) -> pd.DataFrame:
    # SELECT name, population, area FROM World WHERE population >= 25000000 OR area >= 3000000;
    return world[(world["area"] >= 3000000) | (world["population"] >= 25000000)].loc[:, ["name", "population", "area"]]


if __name__ == "__main__":
    world = pd.read_csv("pandas dataset/595. Big Countries.csv")
    print(big_countries(world))

# Done ✅
