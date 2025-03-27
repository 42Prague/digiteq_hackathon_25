import ast
import math
import re
from pathlib import Path

import pandas as pd

EMOJIS_NAMES = ["angry", "crying", "happy", "sad", "surprised"]


def add_one_line(df, file_name, emojis, x_s, y_s):
    one_label = {"file_name": file_name, "moods_pred": emojis, "x_s_pred": x_s, "y_s_pred":
        y_s}
    df = pd.concat([df, pd.DataFrame([one_label])], ignore_index=True)
    return df


def parse_standard_output(captured_output):
    captured_string = captured_output.getvalue()
    print(captured_string)
    captured_string = captured_string.split("\n")
    file_name = ""
    moods = []
    x_s = []
    y_s = []
    predictions = pd.DataFrame(columns=["file_name", "moods_pred", "x_s_pred", "y_s_pred"])
    pictures_history = set()
    for line in captured_string:
        line_source = line
        line = line.split(" ")
        if len(line) >= 2:
            if line[0] == "Picture:":
                if len(moods) > 0:
                    predictions = add_one_line(predictions, file_name, moods, x_s, y_s)
                file_name = line[1]
                if file_name in pictures_history:
                    raise RuntimeError(f"Picture name {file_name} appeared more than once in your output!")
                pictures_history.add(file_name)
                moods = []
                x_s = []
                y_s = []
            if line[0] == "Emoji:":
                if line[1].lower() in EMOJIS_NAMES:
                    moods.append(line[1])
                else:
                    print(
                        f"Invalid emoji name: {line[1]}, possible names {EMOJIS_NAMES}")
                if len(line) == 5:
                    if line[2] == "Coordinates:":
                        try:
                            x_s.append(int(re.search(r'\d+', line[3]).group()))
                            y_s.append(int(re.search(r'\d+', line[4]).group()))
                        except:
                            print(f"Unable to parse coordinates from {line_source}")
        else:
            if line[0]:
                print(f"Missing information in line: {line}")
    if len(moods) > 0:
        predictions = add_one_line(predictions, file_name, moods, x_s, y_s)

    return predictions


def calculate_distance(point_a, point_b):
    return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)


def calculate_row(row):
    # check emojis
    emojis_preds = row["moods_pred"]
    emojis_labels = row["moods"]
    points = 0
    for emoji_pred in set(emojis_preds):
        if emoji_pred in emojis_labels:
            points += min(emojis_preds.count(emoji_pred), emojis_labels.count(
                emoji_pred)) * 0.5
            # check detected distances
            indices_preds = [i for i, x in enumerate(emojis_preds) if x == emoji_pred]
            indices_labels = [i for i, x in enumerate(emojis_labels) if x == emoji_pred]
            for index_label in indices_labels:
                for index_prediction in indices_preds:
                    distance = calculate_distance(
                        [row["x_s"][index_label], row["y_s"][index_label]],
                        [row["x_s_pred"][index_prediction],
                         row["y_s_pred"][index_prediction]])
                    if distance < 40:
                        points += 0.5
                        break
    return points


def calculate_points(predictions, labels):
    merged = pd.merge(predictions, labels, on="file_name")
    merged['points'] = merged.apply(calculate_row, axis=1)
    points = sum(merged['points'])
    print(f"You get {points} points.")
    if points > 100:
        print("Congratulation.")
    return points


def str_to_int_list(a):
    a = a.split(",")
    a = [int(re.search(r'\d+', number).group()) for number in a]
    return a


def load_labels(labels_path):
    labels = pd.read_csv(Path(labels_path), sep=';', index_col=0)
    labels['moods'] = labels.apply(lambda x: ast.literal_eval(x.moods), axis=1)
    labels["y_s"] = labels.apply(lambda x: str_to_int_list(x.y_s), axis=1)
    labels["x_s"] = labels.apply(lambda x: str_to_int_list(x.x_s), axis=1)
    return labels


def evaluate(captured_output, labels_path):
    predictions = parse_standard_output(captured_output)
    if predictions.empty:
        print("No evaluation can be done, no predictions were writen in standard output, "
              "please check you have printed your predictions in proper format.")
        return 0
    else:
        labels = load_labels(labels_path)
        return calculate_points(predictions, labels)
