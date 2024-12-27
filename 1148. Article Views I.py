import pandas as pd


def article_views(views: pd.DataFrame) -> pd.DataFrame:
    # SELECT DISTINCT author_id AS id FROM views WHERE author_id = viewer_id ORDER BY author_id;
    return pd.DataFrame({"id": sorted(views[views["author_id"] == views["viewer_id"]]["author_id"].unique())})


if __name__ == "__main__":
    views = pd.read_csv("pandas dataset/1148. Article Views I.csv")
    print(article_views(views))

# Done ✅
