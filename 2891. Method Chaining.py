import pandas as pd


def findHeavyAnimals(animals: pd.DataFrame) -> pd.DataFrame:
    # return animals[animals['weight'] > 100].sort_values(['weight'], ascending=False)[['name']]
    return animals.loc[
               animals["weight"] > 100,
               ["name", "weight"]
           ].sort_values(
        by="weight",
        ascending=False
    ).iloc[:, :1]


if __name__ == "__main__":
    animals = pd.read_csv("pandas dataset/2891. Method Chaining.csv")
    print(findHeavyAnimals(animals))

# Done ✅
