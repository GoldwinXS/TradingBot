import GetOldTweets3 as got
import pandas as pd


def return_percent(small, large, string):
    try:
        percent = string + ": " + str(round(((small / large) * 100), 2)) + "%"
    except ZeroDivisionError:
        print("Division by 0 error!")
        percent = "error"
    return percent


def get_tweets_by_keyword(keyword, date_from, date_until, num_tweets=10):
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(keyword) \
        .setSince(date_from) \
        .setUntil(date_until) \
        .setMaxTweets(num_tweets) \
        .setLang('English') \
        .setTopTweets(True)
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    return tweets


def get_tweets_by_username(username, date_from, date_until, num_tweets=10):
    # date format = "2015-09-10"
    tweetCriteria = got.manager.TweetCriteria().setUsername(username) \
        .setSince(date_from) \
        .setUntil(date_until) \
        .setMaxTweets(num_tweets)
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)

    return tweets


def list_dates(start, end):
    dates = []
    pd_time = pd.date_range(start=start, end=end)
    for i in range(len(pd_time)):
        dates.append(convertpdtime_to_str(pd_time[i]))
    return dates


def convertpdtime_to_str(pd_timestamp_object):
    sy, sm, sd = pd_timestamp_object.year, pd_timestamp_object.month, pd_timestamp_object.day

    if sm >= 10 and sd >= 10:
        string = str(sy) + '-' + str(sm) + '-' + str(sd)
    elif sm >= 10 and sd < 10:
        string = str(sy) + '-' + str(sm) + '-' + '0' + str(sd)

    elif sm < 10 and sd >= 10:
        string = str(sy) + '-' + '0' + str(sm) + '-' + str(sd)
    else:
        string = str(sy) + '-' + '0' + str(sm) + '-' + '0' + str(sd)

    return string


def list_tweets_btwn_dates_keyword(keyword, start, end, num_tweets=2, verbose=0):
    all_tweets = []
    dlist = list_dates(start=start, end=end)

    for i in range(len(dlist)):
        try:
            all_tweets.append([])
            print(i, all_tweets)
            tweets = get_tweets_by_keyword(keyword, dlist[i - 1], dlist[i], num_tweets=num_tweets)
            for tweet in tweets:
                all_tweets[i].append(tweet.text)
        except IndexError:
            pass

        if verbose == 1 or verbose == 2:
            print(return_percent(i, len(dlist), "Fetching tweets"))
        # time.sleep(0.01)

    return all_tweets

# t_list = list_tweets_btwn_dates_keyword('realDonaldTrump', '2019-07-01', '2019-07-03', num_tweets=100)
# print(t_list)

# tweets = get_tweets_by_keyword('bitcoin', '2019-07-01', '2019-07-02', num_tweets=100)
# print(len(tweets))
# tweets = get_tweets_by_username('realDonaldTrump', '2019-07-01', '2019-07-09', num_tweets=30)
#
# for tweet in tweets:
#     print(tweet.text)

# tweetCriteria = got.manager.TweetCriteria().setQuerySearch('trump, feel').setSince(r'2017-07-01').setUntil(r'2017-12-01').setMaxTweets(10)
# # print(tweetCriteria)
# tweet = got.manager.TweetManager.getTweets(tweetCriteria)

# for tw in tweet:
#     print(tw.text)
