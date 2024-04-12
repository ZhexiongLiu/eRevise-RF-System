import pandas as pd
from erevise.models import ScoreEssay
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix


def get_evaluation(topic, threshold):
    if topic == "MVP":
        path = "../statics/data/MVP/Random Compiled 155 MVP.csv"
        # path2 = "/Users/zhl166/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Code/eRevise/data_raw/N143 MVP Essays_Original_and_Draft.csv"

    elif topic == "SPACE":
        path = "../statics/data/SPACE/Random Compiled 300 Space.csv"
    else:
        raise "wronxg path"

    df = pd.read_csv(path)
    essays = df["first_draft"]
    gold_level = list(df["feedback_level"])
    predict_level = []
    for i in tqdm(range(len(essays))):
        essay = essays[i]
        # essay = """
        # In the story “A Brighter Future”,yes the author convince me that winning the fight against poverty is achievable in our lifetime. People need a home just like animals need a home ,and its just not fair that we have homes and they don’t. It says that “The plan is to get people out of poverty, assure them access to health care and help them stabilize the economy and quality of life in their communities.” Another one is “Villages get technical advice and practical items,such as fertilizer,medicine and school supplies.”Also it says that “It is hard for me to see people sick with preventable diseases, people who are near death when they shouldn’t have to be.I just get scared and sad.” These 3 examples mean that people don’t need to go through poverty , it’s sad and scary for them and they don’t need to go through this.
        # """
        # essay = """
        # In the story “A Brighter Future” yes the author convince me that winning the fight against poverty is achievable in our lifetime. Yes we need to win the fight against poverty because everybody needs a home, shelter, food,and money. It say that “Their crops were dying because they could not afford the necessary fertilizer and irrigation”Another one is that “Its hard for me to see people sick with preventable diseases,people who are near death when they shouldn’t have to be.”Also “.Little kids were wrapped in cloth on their mothers backs,or running around in bare feet and tattered clothing.” These three examples mean that we need to help them have a better life and a better home than the busty,dirty ground.
        # """

        level = gold_level[i]

        essay_score = ScoreEssay(essay, topic=topic, threshold=threshold)
        pred_level = essay_score.get_feedback_level()
        pred_level = pred_level[0][0]
        predict_level.append(pred_level)


        npe = essay_score.get_npe()
        spc = essay_score.get_spc()
        # print(npe)
        # print(spc)
        # print(level, pred_level, npe, spc)

    print(classification_report(gold_level, predict_level, target_names=["level1", "level2", "level3"]))
    print("QWK", cohen_kappa_score(gold_level, predict_level, labels=None, weights= 'quadratic', sample_weight=None))
    print(confusion_matrix(gold_level, predict_level))

# SPACE 0.71 window_size=5
# MVP 0.66 window_size=5

## new results MVP 0.56 with threshold 0.6 with sentence embeddings
## new results SPACE 0.585 with threshold 0.6 with sentence embeddings

for threshold in [0.6, 0.7, 0.8, 0.9]:
    get_evaluation("MVP", threshold)
    print("MVP")

for threshold in [0.6, 0.7, 0.8, 0.9]:
    get_evaluation("SPACE", threshold)
    print("SPACE")