from keras.models import load_model
import tensorflow as tf
import tkinter as tk
from ProjectUtils import *
from TwitterUtils import *
import os
from pandas.plotting import register_matplotlib_converters
from keras.preprocessing.sequence import pad_sequences

register_matplotlib_converters()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # shut tensorflow up

""" SETUP """
##############

padding = 50
dimensions = 100
base_dir = ''

##############


bot = "sentiment_model.h5"  # path to file
loaded_bot = load_model(base_dir+ bot)  # loaded bot object
tokenizer = pd.read_pickle('tokenizer.pickle').values[0][0]

print("Loading glove data.")
gd = GloveLoader("glove.twitter.27B." + str(dimensions) + "d.txt")


def test_predict(single_sentence):
    prepared_sentence = np.array([np.array(gd.tokenize_sentence(single_sentence, pad=padding))])

    prediction = loaded_bot.predict(prepared_sentence)
    print(prediction)

    return prediction


def batch_predict(sentence_list):
    prepared_sentences = []
    for sentence in sentence_list:
        prepared_sentences.append(gd.tokenize_sentence(sentence, pad=padding))

    prepared_sentences = tokenizer.texts_to_sequences(sentence_list)
    paded_sequences = pad_sequences(prepared_sentences,50)
    # prepared_sentences = np.array(prepared_sentences)
    # print(prepared_sentences)
    predictions = loaded_bot.predict(paded_sequences)

    temp = []

    for prediction in predictions:
        temp.append(prediction[1])
    predictions = temp
    return predictions



def predict_from_twitter(keyword, start, end, num_tweets, verbose=0):
    t_list = list_tweets_btwn_dates_keyword(keyword, start, end, num_tweets=num_tweets, verbose=verbose)
    r_list = []
    d_list = []
    i_list = []

    all_dates = list_dates(start, end)

    for i in range(1, len(t_list)):
        r_list.append(average_list(batch_predict(t_list[i])))
        i_list.append(i)
        d_list.append(all_dates[i - 1])
        if verbose == 1:
            print(return_percent(i, len(t_list), 'Grabbing tweets and predicting'))

        if verbose == 2:
            print(return_percent(i, len(t_list), 'Grabbing tweets and predicting'))
            # print(t_list[i])

    dat = pd.DataFrame({'prediction_results': r_list,
                        'index': i_list,
                        'dates': d_list},
                       dtype=float)
    return dat


def predict_from_saved_df(pandas_df, verbose=0, timekey='dates', tweetkey='tweets'):
    t_list = []
    all_dates = []
    num_tweets_list = []
    for i in range(len(pandas_df)):
        t_list.append([pandas_df[tweetkey][i]])
        all_dates.append(pandas_df[timekey][i])
        num_tweets_list.append(len(pandas_df[tweetkey][i]))

    r_list = []
    d_list = []
    p_list = []  # number of tweets per average

    for i in range(0, len(t_list)):
        r_list.append(average_list(batch_predict(t_list[i])))
        d_list.append(all_dates[i])
        p_list.append(num_tweets_list[i])
        if verbose == 1:
            print(return_percent(i, len(t_list), 'Grabbing tweets and predicting'))

        if verbose == 2:
            print(return_percent(i, len(t_list), 'Grabbing tweets and predicting'))
            print(t_list[i])

    dat = pd.DataFrame({'prediction_results': r_list,
                        'dates': d_list,
                        'num_predictions': p_list})
    return dat


# loaded_df = pd.read_pickle('tweets/tweets_bitcoin_500_daily.pickle')
# test = predict_from_saved_df(loaded_df, verbose=1)
# # print(test)
# # print(test['dates'])
# test['Timestamp'] = pd.to_datetime(test['dates'])
# # test['dates'] = pd.to_datetime(test['dates'])
# # plot_df('dates', ['prediction_results'], test)
#
# b_df = pd.read_csv("data/Bitcoin Data.csv", dtype=float, )
# b_df['Timestamp'] = pd.to_datetime(b_df['Timestamp'], unit='s')
#
# df = test.merge(b_df, how='inner', on='Timestamp')
# # df = norm_ops_simple(df)
# # df['Weighted_Price'] = norm_ops_simple(df['Weighted_Price'])
# df['mood_momentum'] = apply_moving_momentum(df, 'prediction_results', window_length=4)
# # df['prediction_results'] = norm_ops_simple(df['prediction_results'])
# # # df['Weighted_Price'] = norm_ops_simple(df['Weighted_Price'])
# # plot_keys = ['prediction_results', 'Weighted_Price', 'mood_momentum']
# # plot_df('Timestamp', plot_keys, df, style='darkmode')
# print(df)
# df = df.reindex(columns=['Timestamp', 'prediction_results', 'mood_momentum', 'High', 'Low', 'Volume_(BTC)', 'Weighted_Price'])
# # print(df)
# df = df.fillna(0)
# df.to_csv('data/crypto_dat_v5.csv')


# print(df.head())
# print(df.info())
# savedf = df[['prediction_results_300', 'High', 'Low', 'Volume_(BTC)', 'Weighted_Price']]
# savedf = pd.concat((savedf, df['High', 'Low', 'Volume_(BTC)', 'Weighted_Price']), axis=1)


# print('Reading some tweets.')
#

target_keyword = 'Bitcoin'
sdate = '2017-11-10'
edate = '2018-02-01'

# results = predict_from_twitter(target_keyword, sdate,edate, num_tweets=300, verbose=1)
# results.to_csv(str(target_keyword)+'_ave300tweets.csv')

#
# results = predict_from_twitter(target_keyword, sdate,edate, num_tweets=100, verbose=2)
# results.to_csv(str(target_keyword)+'_ave100tweets.csv')

infile = 'tweets_bitcoin_100_daily'

in_df = pd.read_pickle(base_dir+'{}.pickle'.format(infile))

for i in range(len(in_df)):

    if type(in_df['tweets'][i]) == float:
        in_df = in_df.drop(i)
        print('OMG, LOOK ITS NAN')

results = predict_from_saved_df(in_df, verbose=1)
results.to_pickle(base_dir+'{}_data_analysis.pickle'.format(infile))


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.wm_title("Twitter Sentiment Bot")

        self.result = tk.StringVar(self.root)

        # Labels
        self.input_label = tk.Label(self.root, text="Put text in here")
        self.input_label.grid(column=0, row=0)

        self.result_label = tk.Label(self.root, textvariable=self.result)
        self.result_label.grid(column=0, row=3)

        # Entry
        self.entry = tk.Entry(self.root, width=75)

        self.entry.grid(column=0, row=1)

        # Submit Button
        self.submit = tk.Button(self.root, text="Submit")
        self.submit.bind("<Button-1>", self.test_predict)
        self.submit.grid(column=0, row=2)

        # Run GUI
        self.root.mainloop()

    def test_predict(self, g):
        sentence = [self.entry.get()]
        # print(sentence)
        # print(type(sentence))
        prediction = test_predict(sentence[0])
        # print(prediction)

        prediction = prediction[0]
        text = "Prediction:"

        if prediction[0] > 0.5:
            text += " Negative   " + "/  Probability: " + str(round(prediction[0] * 100, 2)) + "%"
        else:
            text += " Positive   " + "/  Probability: " + str(round(prediction[1] * 100, 2)) + "%"

        self.result.set(text)

# App()
