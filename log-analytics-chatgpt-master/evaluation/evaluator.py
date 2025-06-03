import pandas as pd
import scipy.special
from nltk.metrics.distance import edit_distance
from sklearn.metrics import accuracy_score
import numpy as np


def evaluate(groundtruth, parsedresult):
    df_groundtruth = pd.read_csv(groundtruth)
    df_parsedlog = pd.read_csv(parsedresult, index_col=False)

    # Remove invalid groundtruth event Ids
    null_logids = df_groundtruth[~df_groundtruth['EventTemplate'].isnull()].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedlog = df_parsedlog.loc[null_logids]

    accuracy_exact_string_matching = accuracy_score(np.array(df_groundtruth.EventTemplate.values, dtype='str'),
                                                    np.array(df_parsedlog.EventTemplate.values, dtype='str'))

    edit_distance_result = []
    for i, j in zip(np.array(df_groundtruth.EventTemplate.values, dtype='str'),
                    np.array(df_parsedlog.EventTemplate.values, dtype='str')):
        edit_distance_result.append(edit_distance(i, j))

    edit_distance_result_mean = np.mean(edit_distance_result)

    (precision, recall, f_measure, accuracy_PA) = get_accuracy(df_groundtruth['EventTemplate'],
                                                               df_parsedlog['EventTemplate'])

    print(
        'Group Accuracy: %.4f, Message-Level Accuracy: %.4f, Edit Distance: %.4f' % (
            accuracy_PA, accuracy_exact_string_matching, edit_distance_result_mean))

    return accuracy_PA, accuracy_exact_string_matching, edit_distance_result_mean


def get_accuracy(series_groundtruth, series_parsedlog, debug=False):
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    real_pairs = 0
    for count in series_groundtruth_valuecounts:
        if count > 1:
            real_pairs += scipy.special.comb(count, 2)

    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    parsed_pairs = 0
    for count in series_parsedlog_valuecounts:
        if count > 1:
            parsed_pairs += scipy.special.comb(count, 2)

    accurate_pairs = 0
    accurate_events = 0  # determine how many lines are correctly parsed
    for parsed_eventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
        error_eventIds = (parsed_eventId, series_groundtruth_logId_valuecounts.index.tolist())
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
                accurate_events += logIds.size
                error = False
        if error and debug:
            print('(parsed_eventId, groundtruth_eventId) =', error_eventIds, 'failed', logIds.size, 'messages')
        for count in series_groundtruth_logId_valuecounts:
            if count > 1:
                accurate_pairs += scipy.special.comb(count, 2)

    precision = float(accurate_pairs) / parsed_pairs
    recall = float(accurate_pairs) / real_pairs
    f_measure = 2 * precision * recall / (precision + recall)
    accuracy = float(accurate_events) / series_groundtruth.size
    return precision, recall, f_measure, accuracy
