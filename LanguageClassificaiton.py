import sys
import math
import pickle
from collections import defaultdict


class Leaf_Node:
    def __init__(self, features):
        self.predictions = label_counts(features)


class Decision_Tree_Node:
    def __init__(self, true_side, false_side, best_vals):
        self.true_side = true_side
        self.false_side = false_side
        self.best_vals = best_vals


class Adaboost_stumps:
    def __init__(self, column_number, left, right, significance):
        self.column_number = column_number
        self.left = left
        self.right = right
        self.significance = significance


def f0_wordlinecontainsenn(line):
    line = line.lower().replace(",", "")
    if "enn" in line:
        return "False"
    return "True"


def f1_wordlinecontainsq(line):
    line = line.lower().replace(",", "")
    if "q" in line:
        return "True"
    return "False"


def f2_wordlinecontainsx(line):
    line = line.lower().replace(",", "")
    if "x" in line:
        return "True"
    return "False"


def f3_wordlinecontainsij(line):
    line = line.lower().replace(",", "")
    if "ij" in line:
        return "False"
    return "True"


def f4_avg_word_length(line):
    line = line.lower().replace(",", "")
    words = line.split(" ")
    length = 0
    for word in words:
        length += len(word)
    avg_length = length / len(words)
    if avg_length > 5:
        return "False"
    return "True"


def f5_wordlinecontains_eng_prep(line):
    line = line.lower().replace(",", "")
    if " a " in line or " an " in line or " the " in line:
        return "True"
    return "False"


def f6_wordlinecontains_dut_prep(line):
    line = line.lower().replace(",", "")
    if " van " in line or " op " in line or " als " in line:
        return "False"
    return "True"


def f7_wordlinecontainsann(line):
    line = line.lower().replace(",", "")
    if "ann" in line:
        return "False"
    return "True"


def f8_wordlinecontainsdan(line):
    line = line.lower().replace(",", "")
    if "dan" in line:
        return "False"
    return "True"


def f9_wordlinecotaints_frequent_dut_words(line):
    ducth_list = ['naar', 'onze', 'deze', 'ons', 'niet', 'ze', 'wij', 'ze', 'er', 'hun', 'be', 'met', 'zo', 'over',
                  'hem', 'weten', 'jouw', 'dan', 'ook', 'hij', 'zijn', 'ik', 'het', 'voor', 'meest']
    line = line.lower().replace(",", "")
    words = line.split(" ")
    for word in words:
        if word in ducth_list:
            return "False"
    return "True"


def f10_wordlinecotaints_frequent_eng_words(line):
    eng_list = ['know', 'your', 'than', 'then', 'for', 'not', 'with', 'he', 'also', 'our', 'these', 'us', 'most', 'to',
                'be', 'i', 'it', 'his', 'they', 'we', 'she', 'there', 'their', 'so', 'about', 'me', 'him']
    line = line.lower().replace(",", "")
    words = line.split(" ")
    for word in words:
        if word in eng_list:
            return "True"
    return "False"


def readTrainData(train_data):
    try:
        train_file = open(train_data, encoding="utf-8-sig")
    except:
        print("Error: File " + train_data + " does not exists")
        exit(0)
    leafnodes = []
    word_lines = []
    for line in train_file.readlines():
        line = line.strip().split("|")
        leafnodes.append(line[0])
        word_lines.append(line[1])
    return word_lines, leafnodes


def readPredictData(test_file, object_file):
    try:
        test_file = open(test_file, 'r')
        object = pickle.load(open(object_file, 'rb'))
    except:
        print("Error: File " + test_file + " does not exists")
        exit(0)
    word_lines = []
    for line in test_file.readlines():
        line = line.strip()
        word_lines.append(line)
    return word_lines, object


def data_extraction(learn_type):
    features = [[] for _ in range(len(word_lines))]
    i = 0
    for word_line in word_lines:
        features[i].append(f0_wordlinecontainsenn(word_line))
        features[i].append(f1_wordlinecontainsq(word_line))
        features[i].append(f2_wordlinecontainsx(word_line))
        features[i].append(f3_wordlinecontainsij(word_line))
        features[i].append(f4_avg_word_length(word_line))
        features[i].append(f5_wordlinecontains_eng_prep(word_line))
        features[i].append(f6_wordlinecontains_dut_prep(word_line))
        features[i].append(f7_wordlinecontainsann(word_line))
        features[i].append(f8_wordlinecontainsdan(word_line))
        features[i].append(f9_wordlinecotaints_frequent_dut_words(word_line))
        features[i].append(f10_wordlinecotaints_frequent_eng_words(word_line))
        if learn_type == "train":
            features[i].append(leafnodes[i])
        i = i + 1
    return features


def calculate_entropy_dt(true_vals, false_vals):
    true_len = len(true_vals)
    false_len = len(false_vals)
    total_true_false = true_len + false_len
    target_counts = dict()
    target_counts["T"] = dict()
    target_counts["F"] = dict()
    for line in true_vals:
        if line[-1] not in target_counts["T"].keys():
            target_counts["T"][line[-1]] = 0
        target_counts["T"][line[-1]] += 1
    for line in false_vals:
        if line[-1] not in target_counts["F"].keys():
            target_counts["F"][line[-1]] = 0
        target_counts["F"][line[-1]] += 1
    entropy_true_value, entropy_false_value = 0.0, 0.0
    for x in target_counts["T"]:
        entropy_true_value += target_counts["T"][x] / sum(target_counts["T"].values())
    for x in target_counts["F"]:
        entropy_true_value += target_counts["F"][x] / sum(target_counts["F"].values())
    entropy = -(true_len / total_true_false) * entropy_true_value \
              - (false_len / total_true_false) * entropy_false_value
    return entropy


def find_true_false_dt(features, best_choice):
    true_vals = []
    false_vals = []
    for row in features:
        if row[best_choice[1]] == best_choice[0]:
            true_vals.append(row)
        else:
            false_vals.append(row)
    return true_vals, false_vals


def find_split_dt(features):
    final_entropy = sys.maxsize
    best_option = []
    targets = dict()
    for col in range(len(features[0]) - 1):
        attributes = ["True", "False"]
        for att in attributes:
            best_choice = []
            best_choice.append(att)
            best_choice.append(col)
            true_vals, false_vals = find_true_false_dt(features, best_choice)
            if len(true_vals) == 0 or len(false_vals) == 0:
                continue
            entropy = calculate_entropy_dt(true_vals, false_vals)
            if final_entropy > entropy:
                final_entropy, best_option = entropy, best_choice
    return final_entropy, best_option

def label_counts(features):
    targets = dict()
    for row in features:
        if row[-1] not in targets.keys():
            targets[row[-1]] = 0
        targets[row[-1]] += 1
    return targets

def build_dt(features):
    # features = [['False', 'False', 'True', 'False', 'False', 'False', 'False', 'True', 'B'],
    #             ['True', 'False', 'True', 'False', 'True', 'False', 'False', 'True', 'B'],
    #             ['True', 'True', 'True', 'False', 'False', 'True', 'True', 'False', 'A'],
    #             ['False', 'True', 'False', 'False', 'True', 'True', 'False', 'True', 'B'],
    #             ['True', 'True', 'True', 'False', 'True', 'False', 'True', 'True', 'B'],
    #             ['False', 'True', 'True', 'False', 'False', 'True', 'False', 'False', 'B'],
    #             ['True', 'True', 'False', 'False', 'True', 'True', 'True', 'True', 'B'],
    #             ['True', 'True', 'True', 'False', 'True', 'False', 'False', 'True', 'B'],
    #             ['True', 'False', 'True', 'False', 'True', 'True', 'False', 'True', 'B'],
    #             ['True', 'False', 'True', 'False', 'True', 'True', 'False', 'True', 'B'],
    #             ['False', 'True', 'True', 'False', 'True', 'False', 'True', 'True', 'B'],
    #             ['False', 'True', 'True', 'True', 'False', 'False', 'True', 'True', 'A'],
    #             ['True', 'False', 'False', 'False', 'True', 'False', 'True', 'True', 'B'],
    #             ['False', 'False', 'True', 'False', 'True', 'True', 'True', 'True', 'B'],
    #             ['False', 'False', 'True', 'False', 'True', 'True', 'False', 'False', 'B'],
    #             ['False', 'True', 'True', 'False', 'False', 'False', 'True', 'True', 'A'],
    #             ['False', 'False', 'True', 'True', 'True', 'True', 'False', 'True', 'B'],
    #             ['False', 'True', 'True', 'True', 'False', 'False', 'True', 'True', 'A'],
    #             ['False', 'True', 'True', 'False', 'False', 'False', 'False', 'True', 'A'],
    #             ['False', 'True', 'True', 'True', 'False', 'False', 'True', 'True', 'A'],
    #             ['True', 'False', 'True', 'False', 'False', 'True', 'True', 'False', 'B'],
    #             ['False', 'True', 'True', 'True', 'False', 'False', 'True', 'True', 'A'],
    #             ['False', 'True', 'True', 'True', 'True', 'False', 'True', 'True', 'A'],
    #             ['True', 'False', 'False', 'False', 'False', 'True', 'True', 'True', 'A'],
    #             ['False', 'True', 'False', 'False', 'True', 'True', 'True', 'True', 'B'],
    #             ['True', 'False', 'True', 'False', 'True', 'False', 'False', 'True', 'B'],
    #             ['False', 'False', 'True', 'False', 'True', 'True', 'True', 'True', 'B'],
    #             ['True', 'True', 'False', 'False', 'True', 'False', 'True', 'True', 'B'],
    #             ['True', 'True', 'False', 'False', 'True', 'True', 'True', 'True', 'B'],
    #             ['True', 'False', 'True', 'False', 'True', 'True', 'True', 'False', 'B'],
    #             ['True', 'False', 'False', 'False', 'False', 'False', 'True', 'True', 'B'],
    #             ['False', 'True', 'True', 'True', 'False', 'True', 'True', 'True', 'A'],
    #             ['True', 'True', 'True', 'True', 'False', 'True', 'True', 'True', 'A'],
    #             ['True', 'True', 'False', 'False', 'False', 'False', 'False', 'True', 'B'],
    #             ['True', 'True', 'False', 'False', 'False', 'False', 'True', 'True', 'B'],
    #             ['False', 'True', 'True', 'False', 'False', 'False', 'False', 'False', 'B'],
    #             ['True', 'True', 'False', 'True', 'False', 'True', 'True', 'False', 'A'],
    #             ['False', 'True', 'True', 'True', 'True', 'False', 'True', 'True', 'A'],
    #             ['False', 'False', 'True', 'False', 'True', 'True', 'False', 'False', 'B'],
    #             ['False', 'True', 'True', 'False', 'False', 'True', 'False', 'True', 'B'],
    #             ['False', 'True', 'True', 'False', 'True', 'False', 'False', 'True', 'B'],
    #             ['False', 'True', 'True', 'False', 'True', 'False', 'True', 'True', 'B'],
    #             ['True', 'False', 'False', 'False', 'True', 'False', 'False', 'True', 'B'],
    #             ['True', 'False', 'True', 'True', 'False', 'False', 'False', 'True', 'A'],
    #             ['True', 'True', 'False', 'False', 'True', 'False', 'False', 'True', 'B'],
    #             ['False', 'True', 'False', 'False', 'True', 'False', 'False', 'False', 'B'],
    #             ['True', 'True', 'True', 'False', 'False', 'True', 'False', 'True', 'B'],
    #             ['True', 'True', 'False', 'True', 'False', 'True', 'True', 'True', 'A'],
    #             ['False', 'True', 'True', 'True', 'True', 'False', 'True', 'True', 'A'],
    #             ['True', 'True', 'True', 'False', 'True', 'False', 'False', 'True', 'B'],
    #             ['True', 'False', 'True', 'False', 'True', 'True', 'False', 'True', 'B'],
    #             ['True', 'False', 'True', 'False', 'True', 'False', 'False', 'True', 'B'],
    #             ['True', 'False', 'True', 'False', 'True', 'True', 'False', 'True', 'B'],
    #             ['False', 'True', 'False', 'True', 'False', 'False', 'True', 'True', 'A'],
    #             ['False', 'True', 'False', 'True', 'True', 'False', 'True', 'True', 'A'],
    #             ['False', 'False', 'True', 'False', 'True', 'False', 'True', 'True', 'A'],
    #             ['True', 'False', 'True', 'False', 'False', 'True', 'False', 'False', 'B'],
    #             ['False', 'True', 'True', 'False', 'True', 'True', 'False', 'True', 'B'],
    #             ['False', 'True', 'True', 'False', 'True', 'False', 'False', 'True', 'B'],
    #             ['True', 'False', 'True', 'False', 'True', 'True', 'False', 'False', 'B'],
    #             ['True', 'True', 'False', 'False', 'False', 'False', 'False', 'True', 'A'],
    #             ['False', 'True', 'True', 'False', 'True', 'False', 'True', 'True', 'A'],
    #             ['True', 'False', 'True', 'True', 'False', 'True', 'False', 'True', 'B'],
    #             ['True', 'False', 'True', 'False', 'True', 'True', 'False', 'True', 'B'],
    #             ['True', 'True', 'True', 'False', 'True', 'False', 'False', 'False', 'B'],
    #             ['False', 'True', 'True', 'False', 'False', 'False', 'True', 'True', 'A'],
    #             ['False', 'False', 'False', 'False', 'True', 'True', 'False', 'True', 'B'],
    #             ['True', 'False', 'True', 'False', 'True', 'True', 'False', 'True', 'B'],
    #             ['True', 'True', 'True', 'True', 'False', 'False', 'False', 'True', 'B'],
    #             ['False', 'True', 'True', 'True', 'False', 'False', 'True', 'False', 'A'],
    #             ['True', 'False', 'True', 'False', 'True', 'True', 'True', 'True', 'B'],
    #             ['False', 'False', 'True', 'True', 'False', 'False', 'True', 'True', 'A'],
    #             ['True', 'True', 'True', 'False', 'False', 'True', 'False', 'True', 'B'],
    #             ['False', 'False', 'False', 'False', 'True', 'True', 'False', 'False', 'A'],
    #             ['True', 'True', 'False', 'False', 'False', 'False', 'True', 'True', 'A'],
    #             ['False', 'False', 'True', 'False', 'True', 'True', 'False', 'False', 'B'],
    #             ['True', 'True', 'True', 'False', 'True', 'True', 'False', 'True', 'B'],
    #             ['False', 'True', 'True', 'False', 'False', 'False', 'True', 'True', 'A'],
    #             ['False', 'False', 'False', 'False', 'True', 'False', 'True', 'True', 'B'],
    #             ['False', 'True', 'True', 'False', 'True', 'False', 'False', 'True', 'B'],
    #             ['False', 'True', 'False', 'False', 'False', 'False', 'False', 'False', 'B'],
    #             ['True', 'True', 'True', 'True', 'False', 'False', 'True', 'True', 'A'],
    #             ['True', 'True', 'True', 'False', 'True', 'False', 'False', 'True', 'B'],
    #             ['True', 'True', 'True', 'True', 'False', 'True', 'True', 'True', 'A'],
    #             ['True', 'False', 'False', 'True', 'True', 'True', 'True', 'True', 'B'],
    #             ['True', 'True', 'False', 'True', 'True', 'False', 'True', 'True', 'B'],
    #             ['False', 'True', 'False', 'False', 'True', 'True', 'False', 'True', 'B'],
    #             ['True', 'False', 'False', 'True', 'True', 'True', 'False', 'False', 'B'],
    #             ['False', 'False', 'True', 'False', 'True', 'True', 'False', 'False', 'B'],
    #             ['False', 'True', 'True', 'False', 'True', 'True', 'True', 'True', 'A'],
    #             ['False', 'False', 'True', 'False', 'True', 'True', 'True', 'True', 'B'],
    #             ['True', 'True', 'True', 'False', 'True', 'True', 'True', 'True', 'B'],
    #             ['False', 'True', 'False', 'True', 'False', 'False', 'True', 'True', 'A'],
    #             ['False', 'False', 'True', 'False', 'True', 'True', 'True', 'False', 'B'],
    #             ['True', 'False', 'True', 'False', 'True', 'True', 'True', 'True', 'B'],
    #             ['True', 'False', 'True', 'True', 'True', 'False', 'False', 'True', 'B'],
    #             ['True', 'True', 'True', 'False', 'False', 'True', 'True', 'True', 'A'],
    #             ['True', 'False', 'False', 'True', 'False', 'False', 'False', 'False', 'A'],
    #             ['False', 'True', 'True', 'True', 'False', 'False', 'True', 'True', 'A'],
    #             ['False', 'True', 'False', 'True', 'False', 'True', 'True', 'True', 'A'],
    #             ['False', 'True', 'False', 'False', 'True', 'False', 'True', 'True', 'A'],
    #             ['False', 'True', 'True', 'False', 'False', 'False', 'True', 'True', 'A'],
    #             ['False', 'False', 'False', 'True', 'True', 'True', 'True', 'True', 'B'],
    #             ['False', 'True', 'True', 'True', 'False', 'True', 'True', 'True', 'A']]
    entropy, best_vals = find_split_dt(features)
    if entropy == sys.maxsize:
        return Leaf_Node(features)
    true_vals, false_vals = find_true_false_dt(features, best_vals)
    true_side = build_dt(true_vals)
    false_side = build_dt(false_vals)
    return Decision_Tree_Node(true_side, false_side, best_vals)


def classify(row, node):
    if isinstance(node, Leaf_Node):
        return node.predictions
    if node.best_vals[0] == row[node.best_vals[1]]:
        return classify(row, node.true_side)
    else:
        return classify(row, node.false_side)


def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


def print_dt(node, spacing=""):
    if isinstance(node, Leaf_Node):
        print(spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print(spacing + str(node.best_vals[0]) + str(node.best_vals[1]))

    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    print_dt(node.true_side, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + '--> False:')
    print_dt(node.false_side, spacing + "  ")


def build_ada(features):
    column_vals = list()
    for x in range(len(features[0]) - 1):
        column_vals.append("col" + str(x))
    features_length = len(features[0]) - 1
    # features_length = 3
    features, sample_weight = find_sample_weight_ada(features)
    object_list = list()
    while features_length > 0:
        best_column, total_error = find_split_ada(features, column_vals)
        significance = find_significance(total_error)
        left, right = find_left_right(features, best_column)
        new_sample_weight_best = sample_weight * math.exp(significance)
        new_sample_weight_non_best = sample_weight * math.exp(-significance)
        features = update_sample_weights(features, best_column, new_sample_weight_best, new_sample_weight_non_best)
        sum = 0
        for row in features:
            sum += row[-1]
        if sum != 1:
            features = normalize_weights(features, sum)
        column_vals[best_column] = "selected_" + str(best_column)
        features_length = features_length - 1
        object_list.append(Adaboost_stumps(best_column, left, right, significance))
    print_object_list(object_list)
    filehandler = open(output_file, 'wb')
    pickle.dump(object_list, filehandler)


def find_sample_weight_ada(features):
    num_rows = len(features)
    for row in features:
        row.append(float(1 / num_rows))
    return features, (1 / num_rows)


def find_true_false_ada(features, col):
    true_rows = []
    false_rows = []
    correct_vals = dict()
    incorrect_vals = dict()
    for row in features:
        if row[col] == "True":
            true_rows.append(row)
        else:
            false_rows.append(row)
    for line in true_rows:
        if line[-2] not in correct_vals.keys():
            correct_vals[line[-2]] = list()
        correct_vals[line[-2]].append(line)
    for line in false_rows:
        if line[-2] not in incorrect_vals.keys():
            incorrect_vals[line[-2]] = list()
        incorrect_vals[line[-2]].append(line)
    return correct_vals, incorrect_vals


def find_split_ada(features, column_vals):
    error_vals = dict()
    for col in range(len(features[0]) - 1):
        # len(features[0]) - 1
        if "selected_" + str(col) not in column_vals:
            correct_vals, incorrect_vals = find_true_false_ada(features, col)
            error_vals[col] = find_total_error_ada(correct_vals, incorrect_vals)
    total_col_error = min(error_vals.items(), key=lambda x: x[1])
    return total_col_error[0], total_col_error[1]


def find_total_error_ada(correct_vals, incorrect_vals):
    total_error = 0.0
    if "nl" in correct_vals.keys():
        for line in correct_vals["nl"]:
            total_error += line[-1]
    if "en" in incorrect_vals.keys():
        for line in incorrect_vals["en"]:
            total_error += line[-1]
    return total_error


def find_significance(total_error):
    significance = 0.5 * math.log((1 - total_error) / total_error)
    return significance


def update_sample_weights(features, best_column, new_sample_weight_best, new_sample_weight_non_best):
    for line in range(len(features)):
        if line == best_column:
            features[line][-1] = new_sample_weight_best
        else:
            features[line][-1] = new_sample_weight_non_best
    return features


def normalize_weights(features, sum):
    for row in features:
        row[-1] = row[-1] / sum
    return features


def find_left_right(features, col):
    true_rows = []
    false_rows = []
    correct_count = dict()
    incorrect_count = dict()
    for row in features:
        if row[col] == "True":
            true_rows.append(row)
        else:
            false_rows.append(row)
    ###
    for line in true_rows:
        if line[-2] not in correct_count.keys():
            correct_count[line[-2]] = 0
        correct_count[line[-2]] += 1
    for line in false_rows:
        if line[-2] not in incorrect_count.keys():
            incorrect_count[line[-2]] = 0
        incorrect_count[line[-2]] += 1
    ###
    left, right = None, None
    left_len, right_len = False, False
    if len(correct_count) > 0:
        left_len = True
        left_max = max(correct_count.values())
        for key in correct_count.keys():
            if correct_count[key] == left_max:
                left = key
                break
    if len(incorrect_count) > 0:
        right_len = True
        right_max = max(incorrect_count.values())
        for key in incorrect_count.keys():
            if incorrect_count[key] == right_max:
                right = key
                break
    if not left_len:
        left = right
    if not right_len:
        right = left
    return left, right


def print_object_list(object_list):
    for obj in object_list:
        print(obj.column_number, obj.left, obj.right, obj.significance)


def predict_ada(object_file, features):
    objects = pickle.load(open(object_file, 'rb'))
    for row in features:
        total_significance_val = 0
        for object in objects:
            if row[object.column_number] == "True":
                total_significance_val += object.significance
            else:
                total_significance_val -= object.significance
        if total_significance_val > 0:
            print("Predicted lang: en")
        else:
            print("Predicted lang: nl")


if __name__ == '__main__':
    if len(sys.argv) == 5:
        if sys.argv[1] == "train":
            data_file = sys.argv[2]
            output_file = sys.argv[3]
            learning_type = sys.argv[4]
            word_lines, leafnodes = readTrainData(data_file)
            features = data_extraction(sys.argv[1])
            if learning_type == "dt":
                dt = build_dt(features)
                print_dt(dt)
                filehandler = open(output_file, 'wb')
                pickle.dump(dt, filehandler)
            elif learning_type == "ada":
                ada = build_ada(features)
            else:
                print("entered learning type is not valid")
        elif sys.argv[1] == "predict":
            object_file = sys.argv[2]
            test_data_file = sys.argv[3]
            learning_type = sys.argv[4]
            if learning_type == "dt":
                word_lines, dt_object = readPredictData(test_data_file, object_file)
                features = data_extraction(sys.argv[1])
                for row in features:
                    print("Actual: %s. Predicted: %s" % (row[-1], print_leaf(classify(row, dt_object))))
            elif learning_type == "ada":
                word_lines, dt_object = readPredictData(test_data_file, object_file)
                features = data_extraction(sys.argv[1])
                predict_ada(object_file, features)
            else:
                print("entered learning type is not valid")
        else:
            print('syntax :train <examples> <hypothesisout> <learningx-type> || syntax :predict <hypothesis> <file> '
                  '<testing-type(dt or ada)>')
    else:
        print('syntax :train <examples> <hypothesisout> <learning-type> || syntax :predict <hypothesis> <file> '
              '<testing-type(dt or ada)>')


