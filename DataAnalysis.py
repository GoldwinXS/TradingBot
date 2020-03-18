from ProjectUtils import *
import math

# df = pd.read_csv("data/crypto_dat_v5.csv", )
# # df = df[int(len(df) / 2):]

bit_df = pd.read_csv('Bitcoin Data.csv', dtype=float)
bit_df['Timestamp'] = pd.to_datetime(bit_df['Timestamp'], unit='s')


#
#
# df['prediction_results_zscore'] = (df['prediction_results']-df['prediction_results'].mean())/df['prediction_results'].std()
# plt.hist(df['prediction_results_zscore'].to_list())
# plt.hist(df['prediction_results'].to_list())
#
#
# plt.show()


# in_df = pd.read_pickle('tweets/tweets_bitcoin_1000_daily.pickle')

# def take_zscore(pandas_df):
#     pandas_df = (pandas_df - pandas_df.mean()) / pandas_df.std()
#     return pandas_df.to_frame()


# def detrend_data(pandas_df):
#     temp = pandas_df.copy()
#     for i in range(0, len(pandas_df)):
#         temp.iloc[i] = pandas_df.iloc[i] - pandas_df.iloc[i - 1]
#     return temp


def correct_with_weights(pandas_df):
    maxx = pandas_df['num_predictions'].max()

    pandas_df = pandas_df.reindex(columns=pandas_df.columns.values.tolist() + ['corrected_data'])
    pandas_df['corrected_data'] = (maxx / pandas_df['num_predictions']) * pandas_df['prediction_results']

    return pandas_df


in_df = pd.read_pickle('tweets_bitcoin_100_daily_data_analysis.pickle')
# print(in_df.info())

dat = correct_with_weights(in_df)
dat['Timestamp'] = pd.to_datetime(dat['dates'])
time_df = pd.to_datetime(dat['dates'])

# plot_df('Timestamp', ['prediction_results', 'num_predictions', 'corrected_data'], dat)
# print(bit_df['Timestamp'])
# print(dat['Timestamp'])

# print(dat.info())
dat = dat.merge(bit_df, how='inner', on='Timestamp')
zdat = dat.reindex(columns=['corrected_data', 'Weighted_Price', 'prediction_results'])

# print(dat.info())
# plot_df('Timestamp', ['corrected_data', 'Weighted_Price'], dat)

# dat['corrected_data', 'Weighted_Price'], n = take_zscore(dat['corrected_data', 'Weighted_Price'])

print(dat.info())
print(dat)
dat['Weighted_Price'] = detrend_data(dat['Weighted_Price'],shift_up=True)
dat['corrected_data'] = detrend_data(dat['corrected_data'],shift_up=True)
dat['prediction_results'] = detrend_data(dat['prediction_results'],shift_up=True)
dat['Open'] = detrend_data(dat['Open'],shift_up=True)
dat['Close'] = detrend_data(dat['Close'],shift_up=True)
dat['High'] = detrend_data(dat['High'],shift_up=True)
dat['Low'] = detrend_data(dat['Low'],shift_up=True)
dat['Volume_(Currency)'] = detrend_data(dat['Volume_(Currency)'],shift_up=True)

# dat = detrend_data(dat)

# zdat, n = take_zscore(zdat)

# dat['Weighted_Price'], n = take_zscore(dat['Weighted_Price'])
zdat = pd.concat((zdat, dat['Timestamp']), axis=1)
# print(zdat.info())
# print(dat)
# print(dat.info())

# plot_df('Timestamp', ['corrected_data', 'Weighted_Price'], zdat)
# print(dat)

# df.info()

# test_df = take_zscore(detrend_data(df['prediction_results'].shift(0)))
# # df['Weighted_Price'] = norm_ops_simple(detrend_data(df['Weighted_Price']))
# test_df['Weighted_Price'] = take_zscore(df['Weighted_Price'])
# test_df['Timestamp'] = pd.to_datetime(df['Timestamp'])
# test_df.info()
#
zdat = zdat[10:]
# zdat['corrected_data'] = zdat['corrected_data'].shift(2)

zdat['difference'] = abs(zdat['corrected_data'] - zdat['Weighted_Price'])
bin_size = 0.1
range_size = int(5 / bin_size)
plt.hist(zdat['Weighted_Price'].to_list(), bins=[x * bin_size for x in range(-range_size, range_size)])
# plt.show()
plt.hist(zdat['corrected_data'].to_list(), bins=[x * bin_size for x in range(-range_size, range_size)])
# plt.show()
plt.hist(zdat['difference'].to_list(), bins=[x * bin_size for x in range(-range_size, range_size)])
# plt.show()
plt.hist(zdat['prediction_results'].to_list(), bins=[x * bin_size for x in range(-range_size, range_size)])
plt.show()
# zdat['difference'] = abs(zdat['corrected_data'] - zdat['Weighted_Price'])
plot_df('Timestamp', ['corrected_data', 'Weighted_Price', 'prediction_results'], zdat)

print(' sum of differences = ', zdat['difference'].sum())
print(' standard deviation of differences = ', zdat['difference'].std())
print(' mean of differences = ', zdat['difference'].mean())


dat.to_csv('crypto_dat_v6.csv')


""" 

 FOR CORRECTED DATA

 SHIFT -1

 sum of differences =  321.88730239158406
 standard deviation of differences =  0.6856670508255238
 mean of differences =  0.9579979237844763
 
 SHIFT 1 

 sum of differences =  323.58969428513046
 standard deviation of differences =  0.7093805589378849
 mean of differences =  0.9630645663247931
 
 SHIFT 0
 
 sum of differences =  312.9787843131452
 standard deviation of differences =  0.6657513889553035
 mean of differences =  0.9287204282289174


 FOR UNCORRECTED DATA
 
 SHIFT -1

 sum of differences =  322.4203306703397
 standard deviation of differences =  0.680161397712457
 mean of differences =  0.9595843174712491
 
 SHIFT 1 

 sum of differences =  319.9478027294311
 standard deviation of differences =  0.7206912444561202
 mean of differences =  0.952225603361402
 
 SHIFT 0
 
 sum of differences =  310.6797878054777
 standard deviation of differences =  0.6826300377741115
 mean of differences =  0.9218984801349487


"""

true_count = 0
total_count = 0
for i in range(len(zdat)):
    try:
        if zdat['corrected_data'][i - 1] < zdat['corrected_data'][i] and zdat['Weighted_Price'][i] < zdat['Weighted_Price'][i + 1]:
            true_count += 1
        if zdat['corrected_data'][i - 1] > zdat['corrected_data'][i] and zdat['Weighted_Price'][i] > zdat['Weighted_Price'][i + 1]:
            true_count += 1
    except IndexError and KeyError:
        pass
    total_count += 1
percent_correct = true_count / total_count
print(percent_correct)

# plot_df('Timestamp', ['prediction_results','Weighted_Price','difference'], test_df)

# plot_df('Timestamp', ['difference'], test_df)
# out_df = test_df.reindex(columns=['Timestamp','prediction_results','Weighted_Price'])
# out_df['Weighted_Price'] = detrend_data(df['Weighted_Price'])
# out_df['Weighted_Price'] = df['Weighted_Price']+abs(df['Weighted_Price'].min())

# print(out_df['Weighted_Price'].min())
# out_df.to_csv('data/data_anal.csv')
# print(out_df)

# count = 0
# temp = []
# index = []
# counter = 0
# for row in in_df['tweets']:
#     print(len(row))
#     temp.append(len(row))
#     index.append(pd.to_datetime(in_df['dates'][counter]))
#     counter += 1
#     for tweet in row:
#         count += 1
#
# plt.plot(index, temp)
# plt.show()
# tlist = []
# print(count)
#
# counter = 0
#
# for i in range(len(in_df['tweets'])):
#     for j in range(len(in_df['tweets'][i])):
#         tlist.append(in_df['tweets'][i][j])
#         counter += 1
#
# temp_df = pd.DataFrame({'tweets': tlist})
# out_df = temp_df.reindex(columns=['Sentiment', 'SentimentText'])
# out_df['SentimentText'] = temp_df['tweets']

# link_list = []

# out_df = out_df[0:10000]
#
# for i in range(len(out_df)):
#     try:
#         print('url tweet deleted')
#         out_df['SentimentText'][i].index('http')
#         out_df = out_df.drop(i)
#     except:
#         print('normal tweet')
#         pass
#     print(return_percent(i, len(out_df), 'deleting url tweets'))
# out_df = out_df.sample(frac=1)
#
# try:
#     out_df = out_df[0:3000]
# except IndexError and KeyError:
#     print('there arent even 3K tweets left :,(')
#
#
# out_df.to_csv('data/bitcoin_labeled_dat_1.csv',index=False)
