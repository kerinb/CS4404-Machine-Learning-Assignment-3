import numpy as np
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score, mean_absolute_error, \
    mean_squared_log_error, median_absolute_error, accuracy_score
import math


def calc_results(y_test_int, pred_int, algo):
    res = "{}, {}, {}, {}, {}, {}, {}, {}, {}\n". \
        format(r2_score(y_test_int,pred_int), explained_variance_score(y_test_int, pred_int),
               mean_absolute_error(y_test_int, pred_int),
               mean_squared_error(y_test_int, pred_int), math.sqrt(mean_squared_error(y_test_int, pred_int)),
               mean_squared_log_error(y_test_int, pred_int), median_absolute_error(y_test_int, pred_int),
               accuracy_score(y_test_int, pred_int), algo)
    text_file = open("Output.txt", 'a')
    text_file.writelines(res)
    text_file.close()


def plot_my_data(y, y_predict, y_vals_on_x, num, colour, algo):
    print('making plot')
    fig, bx = plt.subplots()
    bx.scatter(y_vals_on_x, y_predict, edgecolors=(0, 0, 0), color=colour)
    bx.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    bx.set_xlabel('Measured')
    bx.set_ylabel('Predicted')
    plt.savefig('{} plot {}.png'.format(algo, num))


# Assumed input is a CSV file of numeric data
def load_data_from_csv_file(num_instances, file_loc, seperator=";", skip_header=True):
    with open(file_loc, "r") as input_file:
        if skip_header:
            input_file.readline()
        for i in range(num_instances):
            instance_data_raw = input_file.readline().split(seperator)

            yield np.array(instance_data_raw, dtype=np.float64)


def determine_features_to_use(dataset):
    x, columns_in_data_set = np.array(dataset).shape
    lst = [None] * columns_in_data_set

    for i in range(columns_in_data_set):
        lst[i] = np.array([instance[i] for instance in dataset])

    features_to_use = []
    for i in range(columns_in_data_set - 1):
        r2, p_val = pearsonr(lst[i], lst[11])
        if abs(r2) > 0.001:
            features_to_use.append(i)

    print("Pearson correlation coefficients")
    print("Fixed Acidity: ", pearsonr(lst[0], lst[11]))
    print("Volatile Acidity: ", pearsonr(lst[1], lst[11]))
    print("Citric Acid: ", pearsonr(lst[2], lst[11]))
    print("Residual Sugar: ", pearsonr(lst[3], lst[11]))
    print("Chlorides: ", pearsonr(lst[4], lst[11]))
    print("Free Sulfur Dioxide: ", pearsonr(lst[5], lst[11]))
    print("Total Sulfur Dioxide: ", pearsonr(lst[6], lst[11]))
    print("Density: ", pearsonr(lst[7], lst[11]))
    print("pH: ", pearsonr(lst[8], lst[11]))
    print("Sulphates: ", pearsonr(lst[9], lst[11]))
    print("Alcohol: ", pearsonr(lst[10], lst[11]))

    return features_to_use
