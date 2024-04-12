import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from erevise.models import ScoreEssay
import os
import re
import json
import pandas

def evaluate(essay, topic="MVP"):
    score_essay = ScoreEssay(essay, topic=topic, threshold=0.5)
    level, npe, spc, match_examples, input_topic_mapper = score_essay.get_feedback_level()

    with open(os.path.join("/Users/zhl166/Library/CloudStorage/Dropbox/Code/Marval/", f"statics/data/{topic}/examples_map.json"), "r") as f:
        mapper = json.load(f)
    if topic == "MVP":
        markers = ["GENERAL", "GENERAL", "HOSPITALS", "MALARIA", "FARMING", "SCHOOL", "HOSPITALS", "GENERAL"]
    elif topic == "SPACE":
        markers = ["HELP_PEOPLE", "HELP_PEOPLE", "HELP_ENVIRONMENT", "HELP_ENVIRONMENT", "TANGIBLE_BENEFITS",
                   "TANGIBLE_BENEFITS", "SPIRIT_OF_EXPLORATION", "SPIRIT_OF_EXPLORATION"]
    else:
        raise "wrong topic!"

    used_marker = []
    for i in range(len(match_examples)):
        marker = markers[i]
        phrase_list = match_examples[i]
        phrase_set = set()
        for j in range(len(phrase_list)):
            phrase = " ".join(phrase_list[j])
            phrase_set.add(phrase)
        for phrase in phrase_set:
            if phrase in mapper:
                used_marker.append(marker)

    used_marker = set(used_marker)
    if "GENERAL" in used_marker:
        used_marker.remove("GENERAL")

    count = len(set(used_marker))
    predicted_topics = sorted(list(set(input_topic_mapper.values())))
    color_topics = sorted(list(set(used_marker)))

    miss_npe = set(predicted_topics) - set(color_topics)
    miss_color = set(color_topics) - set(predicted_topics)

    count_list = []
    for topic in ["FARMING", "HOSPITALS", "MALARIA", "SCHOOL"]:
        only_pred_count = 0
        only_color_count = 0
        both_count = 0
        neither_count = 0
        if topic in predicted_topics and topic not in color_topics:
            only_pred_count = 1
        if topic not in predicted_topics and topic in color_topics:
            only_color_count = 1
        if topic in predicted_topics and topic in color_topics:
            both_count = 1
        if topic not in predicted_topics and topic not in color_topics:
            neither_count = 1
        count_list.append([only_pred_count, only_color_count, both_count, neither_count])

    count_str = f"{count_list[0][0]} {count_list[0][1]} {count_list[0][2]} {count_list[0][3]} vs " \
                f"{count_list[1][0]} {count_list[1][1]} {count_list[1][2]} {count_list[1][3]} vs " \
                f"{count_list[2][0]} {count_list[2][1]} {count_list[2][2]} {count_list[2][3]} vs " \
                f"{count_list[3][0]} {count_list[3][1]} {count_list[3][2]} {count_list[3][3]} vs"


    print(f"npe {npe} vs color_num {count} ; {count_str} {predicted_topics} vs {color_topics} ; npe_has_more {miss_npe} vs color_has_more {miss_color}")

    return len(predicted_topics), len(color_topics)

if __name__ == '__main__':
    DRAFT = ["first_draft", "second_draft"]
    for dt in DRAFT:
        TOPIC = ["MVP", "SPACE"]
        for tp in TOPIC:
            if tp == "MVP":
                data = pd.read_csv("/Users/zhl166/Library/CloudStorage/Dropbox/Code/Marval/statics/data/mvp/Random Compiled 155 MVP.csv")
            elif tp == "SPACE":
                data = pd.read_csv("/Users/zhl166/Library/CloudStorage/Dropbox/Code/Marval/statics/data/space/Random Compiled 300 SPACE.csv")
            else:
                raise "wrong topic"
            npe_list = []
            color_list = []
            essays = data[dt].tolist()
            for essay in essays:
                npe_num, color_num = evaluate(essay, topic=tp)
                npe_list.append(npe_num)
                color_list.append(color_num)

            kappa = cohen_kappa_score(npe_list, color_list, weights="quadratic")
            print(f"{tp} kappa", kappa)
            print(confusion_matrix(npe_list, color_list))
