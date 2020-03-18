from twitter_utils import *
import os
import time


day_amt = 10
daily_tweet_amt = 1000
keyword = 'bitcoin'
script_repetitions = 500
wait_time = 10
file_start_date = '2013-01-01'


file = 'tweets'+'_'+keyword+'_'+str(daily_tweet_amt)+"_daily"+'.pickle'
total_tweets = day_amt*daily_tweet_amt*script_repetitions


print('Grabbing '+str(total_tweets)+" tweets...")
for i in range(script_repetitions):
    loaded_df = []
    first_load = False
    print(return_percent(i+1, script_repetitions, 'Scraped')+'  Remaining tweets: '+str(total_tweets-(day_amt*(i+1)*daily_tweet_amt))+'  Total taken: ~' + str(day_amt*(i+1)*daily_tweet_amt))

    if os.path.isfile('tweets/'+file):
        loaded_df = pd.read_pickle('tweets/'+file)
        first_date = loaded_df['dates'].iloc[len(loaded_df) - 1]

    else:
        first_date = file_start_date
        first_load = True
    dates = []
    tweets = []
    tweets_list_str = []
    # keep and save to file as strings because GOT3 works with them
    last_date = convertpdtime_to_str(pd.datetime.fromisoformat(first_date) + pd.Timedelta(days=day_amt))
    days = list_dates(first_date, last_date)
    days = days[1:]
    for i in range(0, len(days)):
        try:
            tweets = get_tweets_by_keyword(keyword, days[i], days[i + 1], num_tweets=daily_tweet_amt)
            dates.append(days[i])
            tweets_list_str.append([])
            for tweet in tweets:
                tweets_list_str[i].append(tweet.text)
        except IndexError:
            pass

    assert len(tweets_list_str[0]) >= 1

    if not first_load:
        save_df = pd.DataFrame({'dates': dates, 'tweets': tweets_list_str})
        loaded_df = pd.concat((loaded_df, save_df), axis=0, sort=False, ignore_index=True)
        loaded_df.to_pickle('tweets/'+file,)
    else:
        save_df = pd.DataFrame({'dates': dates, 'tweets': tweets_list_str})
        save_df.to_pickle('tweets/'+file)


    time.sleep(wait_time)