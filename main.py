import csv

import numpy as np
import itertools
import csv
import random
from sklearn.model_selection import train_test_split


# as data is transaction data we will be reading it directly
def load_groceries_data():
    transactions = []

    with open('data/process-data.csv') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            if any(x.strip() for x in row):
                transactions.append(row)
    return transactions


def load_rating_data():
    rating = dict()
    with open('data/rating.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            if any(x.strip() for x in row):
                rating[row[0]] = float(row[1])
    return rating


def train(transactions):
    # Training step:
    # Calculate OD
    from collections import Counter

    items_list = [i for item in transactions for i in item]

    od_dict = Counter(items_list)
    sorted(od_dict.items(), key=lambda x: x[1]).reverse()

    # Calculate CD
    # Flatten the list of transactions and make all items unique
    unique_items = set(itertools.chain.from_iterable(transactions))

    # Get all combinations of pairs
    all_pairs = list(itertools.combinations(unique_items, 2))

    # Create the dictionary to store pairs
    cd_dict = {pair: len([x for x in transactions if set(pair) <= set(x)]) for pair in all_pairs}

    # Create Rules
    pairwise_association_rules = dict()
    for item, od in od_dict.items():
        from models import new_rule

        associate_items_dict = dict()
        for pair, cd in cd_dict.items():
            if item in list(pair) and cd > 0:
                associate_item = pair[0]
                if associate_item == item:
                    associate_item = pair[1]
                associate_items_dict[associate_item] = cd
                pairwise_association_rules[item] = new_rule(od, associate_items_dict)

    return pairwise_association_rules


def improved_recommend(pairwise_association_rules, inputItems, rating):
    filtered_rules_dict = {k: v for k, v in pairwise_association_rules.items() if k in inputItems}
    for item, value in filtered_rules_dict.items():
        value.associate_items = {k: v for k, v in value.associate_items.items() if k not in inputItems}

    # recommend
    recommend_dict = dict()
    prob_dict = dict()
    weight_dict = dict()
    for k, v in filtered_rules_dict.items():
        for item, cd in v.associate_items.items():
            # calculate probability conditional p
            prob = cd / v.od
            if item in prob_dict:
                prob_dict[item] = prob_dict[item] + prob
            else:
                prob_dict[item] = prob

            # calculate weight
            weight = v.od
            if item in weight_dict:
                weight_dict[item] = weight_dict[item] + weight
            else:
                weight_dict[item] = weight

            # init item to recommend_dict
            recommend_dict[item] = 0

    # calculate rf
    for recommend_item in recommend_dict:
        rf = prob_dict[recommend_item] * weight_dict[recommend_item] * rating[recommend_item]
        recommend_dict[recommend_item] = round(rf * 10) / 10

    sorted_data = {k: v for k, v in sorted(recommend_dict.items(), key=lambda x: x[1], reverse=True)}
    return sorted_data


def original_recommend(pairwise_association_rules, inputItems):
    # filter
    filtered_rules_dict = {k: v for k, v in pairwise_association_rules.items() if k in inputItems}
    for item, value in filtered_rules_dict.items():
        value.associate_items = {k: v for k, v in value.associate_items.items() if k not in inputItems}

    # recommend
    recommend_dict = dict()
    prob_dict = dict()
    weight_dict = dict()
    for k, v in filtered_rules_dict.items():
        for item, cd in v.associate_items.items():
            # calculate probability conditional p
            prob = cd / v.od
            if item in prob_dict:
                prob_dict[item] = prob_dict[item] + prob
            else:
                prob_dict[item] = prob

            # calculate weight
            weight = v.od
            if item in weight_dict:
                weight_dict[item] = weight_dict[item] + weight
            else:
                weight_dict[item] = weight

            # init item to recommend_dict
            recommend_dict[item] = 0

    # calculate rf
    for recommend_item in recommend_dict:
        rf = prob_dict[recommend_item] * weight_dict[recommend_item]
        recommend_dict[recommend_item] = round(rf * 10) / 10
    sorted_data = {k: v for k, v in sorted(recommend_dict.items(), key=lambda x: x[1], reverse=True)}

    return sorted_data


def calculate_precision_recall(recommendation, expected_correct_prediction, top_n):
    total_prediction = len(recommendation)
    if total_prediction == 0:
        return 0, 0
    amount_expected_correct_prediction = len(expected_correct_prediction)
    actual_correct_item = 0
    top_n_item = sorted(recommendation, key=recommendation.get, reverse=True)[:top_n]
    for item in expected_correct_prediction:
        if item in top_n_item:
            actual_correct_item += 1
    p = actual_correct_item / total_prediction
    r = actual_correct_item / amount_expected_correct_prediction

    return p, r,actual_correct_item


if __name__ == "__main__":
    # CÁC BIẾN SỐ CẦN SỬ DỤNG CHẠY GIẢI THUẬT
    k_fold = 10

    data = load_groceries_data()
    rating_data = load_rating_data()

    for i in range(k_fold):
        train_data, test_data = train_test_split(data, test_size=0.1, train_size=0.9, random_state=1, shuffle=True)
        r_dict = dict()
        r_dict["p"] = 0
        r_dict["r"] = 0
        r_dict["improvedP"] = 0
        r_dict["improvedR"] = 0

        # Train the data
        rules = train(train_data)

        for transaction in test_data:
            input_items = random.sample(transaction, 2)
            expected_prediction = np.setdiff1d(transaction, input_items)

            if len(expected_prediction) == 0:
                continue

            # Find sorted recommendation
            sorted_recommendation = original_recommend(rules, input_items)

            # Find sorted recommendation
            sorted_improved_recommendation = improved_recommend(rules, input_items, rating_data)

            # Calculate precision, recall
            precision, recall = calculate_precision_recall(sorted_recommendation, expected_prediction, 5)

            # Calculate precision, recall
            improved_precision, improved_recall = calculate_precision_recall(sorted_improved_recommendation,
                                                                             expected_prediction, 5)

            r_dict["p"] = r_dict["p"] + precision
            r_dict["r"] = r_dict["r"] + recall
            r_dict["improvedP"] = r_dict["improvedP"] + improved_precision
            r_dict["improvedR"] = r_dict["improvedR"] + improved_recall

        r_dict["p"] = r_dict["p"] / len(test_data)
        r_dict["r"] = r_dict["r"] / len(test_data)
        r_dict["improvedP"] = r_dict["improvedP"] / len(test_data)
        r_dict["improvedR"] = r_dict["improvedR"] / len(test_data)

        print("Result of fold ", i)
        print(r_dict)
        print("-----------------------------")
