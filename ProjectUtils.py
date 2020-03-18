import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import csv
from string import punctuation
import re
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


def plot_df(x_key, y_key_list, some_df, mark_keys=None, style=None):

    mark_keys = mark_keys
    styles = ['dark_background', 'ggplot', 'fivethirtyeight']
    colours = ['xkcd:steel blue', 'xkcd:emerald', 'xkcd:raspberry', 'xkcd:golden yellow', 'xkcd:dark green', 'y', 'k']
    markers = ['x', '+', '1', '2', '3', "4"]

    if style:
        if style == 'darkmode':
            plt.style.use(styles[0])
        if style == 'niceplot':
            plt.style.use(styles[0])
        if style == '538':
            plt.style.use(styles[0])

    if not mark_keys:
        for i in range(0, len(y_key_list)):
            plt.plot(x_key, y_key_list[i], data=some_df, color=colours[i], label=y_key_list[i], )

    if mark_keys:
        index = 0
        for i in range(0, len(y_key_list + mark_keys)):
            if i < len(y_key_list):
                plt.plot(x_key, y_key_list[i], data=some_df, color=colours[i], label=y_key_list[i], )
            else:
                plt.plot(x_key, mark_keys[index], data=some_df, color=colours[i], label=mark_keys[index],
                         marker=markers[index])
                index += 1
    plt.legend(loc="best")
    plt.show()


def return_percent(small, large, string):
    try:
        percent = string + ": " + str(round(((small / large) * 100), 2)) + "%"
    except ZeroDivisionError:
        print("Division by 0 error!")
        percent = "error"
    return percent


# TODO: average over first few numbers better (currently just kept the same)
def apply_moving_momentum(pandas_df, key, window_length):
    aves = []
    results = None

    for i in range(len(pandas_df)):
        velocity = 1
        mass = pandas_df[key][i]

        if i >= window_length:
            velocity = pandas_df[key][i] - pandas_df[key][i - 1]
            mass = pandas_df[key].iloc[i - window_length:i].to_frame().mean().values[0]
            # print(type(velocity),type(mass))
        aves.append(mass * velocity)
        results = pd.DataFrame({key + '_P': aves})

    return results


def prepare_data(pandas_df, time_window_size, verbose=0):
    len_df = len(pandas_df)
    num_features = pandas_df.shape[1]
    num_time_batches = len_df - time_window_size
    print("Number of time batch splits: " + str(num_time_batches))
    inputs = np.empty((num_time_batches, time_window_size - 1, num_features))
    labels = np.empty((num_time_batches, 1))
    normalization_factors = np.empty((num_time_batches, 4, num_features))
    for i in range(0, len_df - time_window_size, ):
        temp = pandas_df.iloc[i:i + time_window_size, :]
        maxx, minx, mean, = temp.max(), temp.min(), temp.mean(),  # get important values
        temp = (temp - mean) / (maxx - minx)  # normalize all inputs between -0.5 and 0.5
        temp = temp * 2  # multiply by 2 to scale normalization between -1 and 1
        std = temp.std()  # get modified std (done later so it is 1 after other steps)
        temp = temp / temp.std()  # scale standard deviation to 1
        # print(i)
        normalization_factors[i][:] = np.array([
            maxx.to_list(),
            minx.to_list(),
            mean.to_list(),
            std.to_list(),
        ])  # need this later to denormalize

        labels[i][:] = temp.iloc[-1:, temp.shape[1] - 1:temp.shape[1]].values
        inputs[i][:] = temp.iloc[:-1, :].values
        if verbose == 1: print(return_percent(i, num_time_batches, "Preparing data"))

    normalization_factors = normalization_factors[:, 0:4,
                            num_features - 1]  # grab max, min, std from numpy array for weighted price (index 8)
    normalization_factors = pd.DataFrame({"max": normalization_factors[:, 0],
                                          "min": normalization_factors[:, 1],
                                          "mean": normalization_factors[:, 2],
                                          'std': normalization_factors[:, 3]})
    return inputs, labels, normalization_factors


def norm_ops(pandas_df):
    # len_df = len(pandas_df)

    norm_factors = []

    maxx, minx, = pandas_df.max(), pandas_df.min(),  # get important values
    norm_factors.append(maxx)
    norm_factors.append(minx)

    pandas_df = (pandas_df - pandas_df.min()) / (pandas_df.max() - pandas_df.min())

    return pandas_df, norm_factors


def norm_ops_simple(pandas_df, ):
    pandas_df = (pandas_df - pandas_df.min()) / (pandas_df.max() - pandas_df.min())
    return pandas_df


def norm_other_data(pandas_df, norm_factors):
    maxx = norm_factors[0]
    minx = norm_factors[1]

    pandas_df = (pandas_df - minx) / (maxx - minx)

    return pandas_df


def denorm_ops(pandas_df, norm_factors, ):
    maxx = norm_factors[0]
    minx = norm_factors[1]

    pandas_df = (pandas_df * (maxx - minx)) + minx

    return pandas_df


def norm_single_data(x, n, target_column):
    normalization = []
    for i in range(2):
        normalization.append(n[i][target_column])
    # print(normalization)

    val = norm_other_data(pd.DataFrame({'x': [x]}), norm_factors=normalization)
    # print(val)
    return val.values[0][0]


def take_zscore(pandas_df):
    norm_factors = []
    norm_factors.append(pandas_df.mean())
    norm_factors.append(pandas_df.std())
    pandas_df = (pandas_df - pandas_df.mean()) / pandas_df.std()

    return pandas_df, norm_factors


def zscore_other_data(pandas_df, norm_factors):
    meanx = norm_factors[0]
    stdx = norm_factors[1]

    return (pandas_df - meanx) / stdx


def take_single_zscore(x, norm_factors, target_column):
    normalization = []
    for i in range(2):
        normalization.append(norm_factors[i][target_column])
    val = zscore_other_data(pd.DataFrame({'x': [x]}), norm_factors=normalization)

    return val.values[0][0]

def undo_zscore(pandas_df,norm_factors):
    meanx = norm_factors[0]
    stdx = norm_factors[1]

    temp = (pandas_df*stdx) + meanx

    return temp


def undo_single_zscore(x, norm_factors, target_column):
    normalization = []
    for i in range(2):
        normalization.append(norm_factors[i][target_column])

    val = undo_zscore(pd.DataFrame({'x': [x]}), norm_factors=normalization, )
    return val.values[0][0]

def detrend_data(pandas_df,shift_up = False):
    temp = pandas_df.copy()
    for i in range(0, len(pandas_df)):
        temp.iloc[i] = pandas_df.iloc[i] - pandas_df.iloc[i - 1]

    if shift_up:
        temp = temp+abs(temp.min())
    return temp


def denorm_single_data(x, n, target_column):
    normalization = []
    for i in range(2):
        normalization.append(n[i][target_column])
    # print(normalization)
    val = denorm_ops(pd.DataFrame({'x': [x]}), norm_factors=normalization, )
    return val.values[0][0]


# test_range = '(-0.5,0.5)'

# df = pd.DataFrame({"data0": [1, 2, 3, 4, 5], "data1": [4, 2, 123, 4, 1]})


def shuffle_two_arrays(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a, b


def remove_duplicate_entires(pandas_df, key):
    string = ''
    try:
        for i in range(len(pandas_df)):
            string = pandas_df[key][i]
            for j in range(i, len(pandas_df)):
                if string == pandas_df[key][j]:
                    pandas_df.drop(pandas_df.index[j])
    except IndexError:
        pass
    return pandas_df


def relu_fnc(x):
    return np.maximum(x, 0, dtype=float)


total_minutes_of_data = 2099760
minutes_per_year = 525600


# TODO: make dynamic plots = FALSE work
def live_plot(x_key, y_key_list, some_df, style=None, speed=0.1, dynamic_plot=True):
    styles = ['dark_background', 'ggplot', 'fivethirtyeight']
    colours = ['xkcd:steel blue', 'xkcd:emerald', 'xkcd:raspberry', 'xkcd:steel grey', 'xkcd:barney', 'y', 'k']

    if style:
        if style == 'darkmode':
            plt.style.use(styles[0])
        if style == 'niceplot':
            plt.style.use(styles[0])
        if style == '538':
            plt.style.use(styles[0])

    df2 = some_df
    plt.ion()

    fig, ax = plt.subplots()
    colour_index = 0

    df2.plot(x=x_key, y=y_key_list[0], ax=ax, color=colours[colour_index])
    if dynamic_plot:
        df2 = some_df[0:0]
        i = 0
        while i <= len(some_df):

            df2 = df2.append(some_df[i:i + 1])
            ax.clear()
            for j in range(len(y_key_list)):
                df2.plot(x=x_key, y=y_key_list[j], ax=ax, color=colours[colour_index])
                colour_index += 1
                plt.draw()
            plt.pause(speed)
            colour_index = 0
            # ax.xlim(0, 100)

            i += 1

        plt.show()
        plt.pause(0.001)


class GloveLoader():

    def __init__(self, path):
        self.words = self.read_glove_model(path)
        self.exclude = set(r'$%&\()*+-/;<=>[\\]^_`{|}~')

    def read_glove_model(self, path):
        words = pd.read_csv(path,
                            sep=" ",
                            index_col=0,
                            header=None,
                            na_values=None,
                            keep_default_na=False,
                            quoting=csv.QUOTE_NONE)

        return words

    def get_word_vector(self, word_string):
        ans = self.words.loc[word_string]
        return ans

    def get_word_index(self, word_string):
        ans = self.words.index.get_loc(word_string)
        return ans

    def clean_string(self, s):
        s = str(s)
        s += " "
        s = re.sub(r"http\S+", " uniformresourcelocator ", s)
        s = s.replace('... ', ' elipsespoints ')
        s = s.replace(' :) ', ' smileyface ')
        s = s.replace(' :D ', ' smileyface ')

        temp = ''
        for i in range(len(s)):
            bad_chars = ['.', '!', '?', ",", '#', r'"']
            if s[i] in bad_chars:
                try:
                    if s[i + 1] in bad_chars or s[i - 1] in bad_chars:
                        s = s[:i] + ' ' + s[i + 1:]
                except IndexError:
                    pass
            temp += s[i]

        s = temp

        temp = ''.join(ch for ch in s if ch not in self.exclude)
        temp = temp.lower()

        s = temp
        s = s.replace('.', ' . ')
        s = s.replace(':', ' : ')
        s = s.replace('@', ' @ ')
        s = s.replace('!', ' ! ')
        s = s.replace('?', ' ? ')
        s = s.replace(',', ' , ')
        s = s.replace('#', ' <hashtag> ')
        s = s.replace('"', ' " ')
        s = s.replace("'", " ' ")
        s = s.replace("smileyface", " <smile> ")
        s = s.replace("smileyface", " <smile> ")
        s = s.replace("uniformresourcelocator", " <url> ")
        s = s.replace("elipsespoints", " ... ")

        return s

    def tokenize_sentence(self, string, pad=None):
        string = self.clean_string(string)
        split_sentence = string.split()
        if pad:
            tokenized_sentence = np.zeros(pad)
        else:
            tokenized_sentence = np.zeros(len(split_sentence))

        for i in range(len(split_sentence)):
            try:
                try:
                    tokenized_sentence[i] = (self.get_word_index(split_sentence[i]))

                except KeyError:
                    tokenized_sentence[i] = self.words.shape[0] + 1
            except IndexError:
                pass
        return np.array(tokenized_sentence)

    def tokenize_data(self, pandas_df, pad=15):
        self.tokenized_data = pandas_df.apply(self.tokenize_sentence, pad=pad)

        return self.tokenized_data

    def dense_to_one_hot(self, label, num_classes=10):
        labels_dense = np.array([label])
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def create_one_hot_labels(self, pandas_df, num_classes=10):

        self.one_hot_labels = pandas_df.apply(self.dense_to_one_hot, num_classes=num_classes)

        return self.one_hot_labels

    def convert_to_tensor(self, data):
        shape = (data.shape[0],) + tuple([data.iloc[0].shape[0]])
        temp = np.zeros(shape)
        for i in range(len(data)):
            temp[i] = data.iloc[i]

        return temp

    def convert_to_simple_tensor(self, data):
        shape = (data.shape[0],) + tuple([data.iloc[0].shape[1]])
        temp = np.zeros(shape)
        for i in range(len(data)):
            temp[i] = data.iloc[i][0]

        return temp



# dataset_path = "./input/Sentiment Analysis Dataset.csv"
# DATASET_ENCODING = "ISO-8859-1"
# input_length = 30
# print("Opening file:", dataset_path)
# df = pd.read_csv(dataset_path,
#                  error_bad_lines=False,
#                  encoding=DATASET_ENCODING,
#                  dtype={"ItemID": int, "Sentiment": int, "SentimentSource": str, "SentimentText": str, },
#                  nrows=1000)
# print("Dataset size (number of tweets):", len(df))
# docs = df["SentimentText"]
# sentiments = df["Sentiment"]
# gd = glove_loader("./glove.twitter.27B/glove.twitter.27B.25d.txt")
# tokenized_data = gd.tokenize_data(docs, pad=input_length)  # .to_frame().to_numpy()
# one_hot_labels = gd.create_one_hot_labels(sentiments, num_classes=2)  # .to_frame().to_numpy()
# print(one_hot_labels)
# print(one_hot_labels.shape)
#
# labels = gd.convert_to_tensor(one_hot_labels)
# print(labels.shape)
# inputs = gd.convert_to_tensor(tokenized_data)
# print(inputs.shape)

def average_list(list):
    div = len(list)
    sum = 0
    list_ave = []
    for elem in list:
        sum += elem

    ave = sum / div
    return ave


def dense_to_one_hot(label, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    labels_dense = np.array([label])
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot[0]
