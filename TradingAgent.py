from ProjectUtils import *
from gym import spaces
from gym.utils import seeding

import numpy as np
from math import floor
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, LSTM
from keras.optimizers import Adam
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

ENV_NAME = "TEST"

""" LOAD DATA """

# df = pd.read_pickle("data/crypto_dat_v4.pickle", )
df = pd.read_csv('crypto_dat_v6.csv')

# print(df)

# df['timestamp'] = df['dates']
# df = df.drop('dates',axis=1)
df = df[np.isfinite(df['Weighted_Price'])]

neuron_multiplier = 10

# TODO reorg max and min length to be in a number of days and not a fraction
days_length_of_training_episode = 30
training_range_percent = 0.775
print(len(df))
training_stop_index = int(len(df) * training_range_percent)
max_training_session_length = len(df) - training_stop_index


# df = df[500:]


class Stocks(gym.Env):

    def __init__(self, pandas_df):
        plt.style.use('dark_background')

        """ PROCESS DATA """
        other_data_keys = ['corrected_data','High','Low','Volume_(BTC)','Volume_(Currency)','Open','Close']
        self.price_keys = ['Weighted_Price', ]
        self.internal_tracking_keys = ['Working Balance', 'Bitcoin Balance', 'Bank Balance', 'Total Worth', 'Transferable Money', 'previous_Weighted_Price', 'previous_prediction_results', 'Baseline']

        # if isinstance(type(df['Timestamp'][0]),type('')):
        #     self.times_df = pd.to_datetime(df['Timestamp'])
        # else:
        self.times_df = pd.to_datetime(df['Timestamp'])

        input_columns = other_data_keys + self.price_keys + self.internal_tracking_keys

        self.df_inputs = pandas_df.reindex(columns=input_columns)
        # self.df_inputs = self.df_inputs.drop('Timestamp', axis=1)

        # print(self.df_inputs.columns)
        self.df_inputs.loc[-1] = np.zeros((len(self.df_inputs.columns.to_list())))
        # self.inputs, self.norms = norm_ops(self.df_inputs)
        self.inputs, self.norms = take_zscore(self.df_inputs)
        self.inputs = self.inputs.to_numpy()  # convert to n
        # umpy array
        # print(self.inputs)
        # print(self.inputs)

        """ DEFINE ACTION AND OBSERVATION SPACE """
        # Set number of actions (buy/sell for coins as well as hold/transfer)
        self.range = 1
        self.num_actions = len(self.price_keys) * 2 + 2
        self.action_space = spaces.Box(low=np.zeros((self.num_actions)), high=np.ones((self.num_actions)))  # , shape=(self.num_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=self.range, shape=[self.df_inputs.shape[1]], dtype=np.float32)
        self.action_space.n = self.num_actions

        """ DEFINE A DICTIONARY TO FIND CORRECT INPUTS BY NAME LATER ON """
        self.key_df_column_dict = {}
        for i in range(len(input_columns)):
            self.key_df_column_dict.update({input_columns[i]: self.df_inputs.columns.get_loc(input_columns[i])})

        """ DEFINE UNCHANGING ENV VARS """
        self.trans_fee = 1
        self.transaction_fee = self.norm(self.trans_fee)
        self.total_step_counter = 0

        """ SET VARS FOR RENDER FUNCTION """
        self.render_keys = self.price_keys + ['Working Balance', 'Bitcoin Balance', 'Bank Balance', 'Total Worth', 'Transferable Money', 'Baseline']  # self.internal_tracking_keys #[:len(self.internal_tracking_keys)-2]

        # find what key corresponds to which index for render fnc

        self.render_df = self.create_empty_render_df()
        self.colours = ['xkcd:ultramarine',
                        'xkcd:vivid purple',
                        'xkcd:gold',
                        'xkcd:irish green',
                        'xkcd:dark pink',
                        'xkcd:lighter green',
                        'xkcd:sea blue']

        """ START ENV """
        self.seed()

        # print(input_columns)
        # print(self.inputs)

    """ CORE FUNCTIONS """

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """ MOST IMPORTANT FUNCTION FOR TIMING"""
        # print(action)
        """ PARSE ACTION """
        assert type(action) == np.ndarray
        action = list(action)
        # for i in range(len(action)):
        #     if action[i] < 0:
        #         action[i]= 0
        #     else:
        #         pass
        # action = [abs(x) for x in action]
        # print(action)
        self.chosen_action = int(np.argmax(action))
        chosen_action_amt = action[self.chosen_action]

        """ REWARD """
        # reward has 4 terms:
        # 1. the total worth compared to initial worth
        # 2. the amount in the bank compared to initial worth
        # 3. a punishment factor for not picking a new action
        # 4. a constant to shift the rewards into a range near 0 at the start
        # 5. a small bonus for making a new choice
        self.reward = self.denorm(self.total_worth) / self.denorm(self.baseline_worth) + (self.denorm(self.bank) / self.denorm(self.random_balance)) - 1  # - np.exp(self.same_actions_counter * 0.01) + 1
        # if self.same_actions_counter ==0:
        #     self.reward +=  np.exp(0.1)
        # print(' same action = ', self.same_actions_counter )

        """ UPDATE STATE """
        coin_values = []
        for i in range(len(self.coin_amounts)):
            coin_values.append(self.inputs[self.step_index][self.df_inputs.columns.get_loc(self.price_keys[i])])

        self.update_internal_data()
        self.process_action(self.chosen_action, chosen_action_amt, coin_values)
        self.time_rewards()
        self.step_index += 1

        """ DEFINE HOW ACTIONS WORK TO AGENT """
        coin_values = []
        for i in range(len(self.coin_amounts)): coin_values.append(self.inputs[self.step_index][self.df_inputs.columns.get_loc(self.price_keys[i])])

        for i in range(len(self.price_keys)):  self.coin_values[i] = self.norm(self.coin_amounts[i] * self.denorm(coin_values[i]))

        self.baseline_worth = self.norm(self.baseline_amount * self.denorm(coin_values[0]))

        """ ADD INTERNAL DATA TO NEXT INPUT """

        self.state = self.inputs[self.step_index]
        self.assess_if_done()
        self.save()

        return self.state, self.reward, self.done, {}

    def reset(self):
        return self.new_start()

    def render(self, mode='live', title=None, **kwargs):

        plt.ion()
        plt.clf()
        self.lists_to_plot = []
        for i in range(len(self.render_keys)):
            self.lists_to_plot.append([])
            self.lists_to_plot[i] = self.render_df[self.render_keys[i]].iloc[:self.step_index].to_list()
        self.ylists = self.lists_to_plot
        self.xlist = self.times_df.iloc[:self.step_index].to_list()
        for i in range(len(self.render_keys)):  # Graph stuff
            graph = plt.plot(self.xlist, self.ylists[i], label=self.render_keys[i], color=self.colours[i])[0]
            graph.set_ydata(self.ylists[i])
            graph.set_xdata(self.xlist)

        plt.xlim(self.xlist[self.render_index] - pd.Timedelta('30 days'), self.xlist[self.render_index])
        plt.legend(loc="best")
        plt.draw()
        plt.pause(.00001)

        self.render_index += 1

    """ CLASS UTILITY FUNCTIONS """

    def assess_if_done(self):

        if undo_single_zscore(self.bank, self.norms, self.key_df_column_dict[self.price_keys[0]]) < 0:
            self.done = True

        # if self.same_actions_counter >= 7:  # must make a new action every 1 week(s)
        #     self.done = True

    def new_start(self):
        self.step_index = np.random.randint(0, training_stop_index)
        self.start_index = self.render_index = self.step_index
        self.random_balance = np.random.randint(100, 2000)
        money_frac = np.random.random(1)[0]
        coin_frac = 1 - money_frac

        self.initial_balance = self.random_balance * money_frac  # 10000  # self.df_inputs[self.price_keys[0]][self.step_index+3]  # - 2300

        self.balance = self.norm(self.initial_balance)
        self.start_balance = self.balance  # this is to track reward

        self.actions_taken_list = []
        self.same_actions_counter = 0
        self.action_index = 0
        self.bank = self.norm(0)
        frac_coin_splits = (self.random_balance * coin_frac) / len(self.price_keys)

        self.coin_amounts = np.zeros((len(self.price_keys)))
        self.coin_values = np.zeros((len(self.price_keys)))
        for i in range(len(self.coin_values)):
            self.coin_values[i] = self.norm(frac_coin_splits)
            self.coin_amounts[i] = frac_coin_splits / self.denorm(self.inputs[self.step_index][self.key_df_column_dict[self.price_keys[i]]])

        self.baseline_worth = self.norm(self.random_balance)
        self.baseline_amount = self.random_balance / self.denorm(self.inputs[self.step_index][self.key_df_column_dict[self.price_keys[0]]])

        self.done = False
        self.render_df = self.create_empty_render_df()

        self.transferable = self.norm(0)
        self.chosen_action = 0
        self.state = self.inputs[self.step_index]
        self.update_internal_data()

        return self.state

    def update_internal_data(self):
        self.total_worth = self.norm(self.denorm(self.balance) + self.denorm(self.coin_values.sum()))

        self.inputs[self.step_index][self.key_df_column_dict[self.internal_tracking_keys[0]]] = self.balance
        self.inputs[self.step_index][self.key_df_column_dict[self.internal_tracking_keys[1]]] = self.coin_values[0]
        self.inputs[self.step_index][self.key_df_column_dict[self.internal_tracking_keys[2]]] = self.bank
        self.inputs[self.step_index][self.key_df_column_dict[self.internal_tracking_keys[3]]] = self.total_worth
        self.inputs[self.step_index][self.key_df_column_dict[self.internal_tracking_keys[4]]] = self.transferable

        try:
            self.inputs[self.step_index][self.key_df_column_dict[self.internal_tracking_keys[5]]] = self.inputs[self.step_index - 1][self.key_df_column_dict[self.internal_tracking_keys[5]]]
            self.inputs[self.step_index][self.key_df_column_dict[self.internal_tracking_keys[6]]] = self.inputs[self.step_index - 1][self.key_df_column_dict[self.internal_tracking_keys[5]]]
        except IndexError:
            self.inputs[self.step_index][self.key_df_column_dict[self.internal_tracking_keys[5]]] = self.inputs[self.step_index][self.key_df_column_dict[self.internal_tracking_keys[5]]]
            self.inputs[self.step_index][self.key_df_column_dict[self.internal_tracking_keys[6]]] = self.inputs[self.step_index][self.key_df_column_dict[self.internal_tracking_keys[5]]]

        self.inputs[self.step_index][self.key_df_column_dict[self.internal_tracking_keys[7]]] = self.baseline_worth

        """ ADD DATA TO RENDER FUNCTION """
        dat = []

        for i in range(len(self.render_keys)):
            dat.append(self.denorm(self.state[self.key_df_column_dict[self.render_keys[i]]]))
        self.render_df.iloc[self.step_index] = dat
        # print(self.render_df.iloc[self.step_index])

    def time_rewards(self):
        if self.action_index > 2:
            a = self.actions_taken_list[self.action_index - 1]
            b = self.actions_taken_list[self.action_index]
            if a == b:
                self.same_actions_counter += 1
            else:
                self.same_actions_counter = 0

        self.action_index += 1

    def denorm(self, x):
        return undo_single_zscore(x, self.norms, self.key_df_column_dict[self.price_keys[0]])

    def norm(self, x):
        return take_single_zscore(x, self.norms, self.key_df_column_dict[self.price_keys[0]])

    def process_action(self, chosen_act, chosen_amt, coin_vals):
        # print('transacion fees: ')
        # print(self.transaction_fee)
        # print(undo_single_zscore(self.transaction_fee,self.norms,self.df_inputs.columns.to_list().index(self.price_keys[0]), ))
        chosen_action_amt = chosen_amt
        chosen_action = chosen_act
        chosen_coin = 0
        has_coins = False
        has_money = False
        self.buy_action = False
        self.sell_action = False
        self.transfer_action = False
        self.money_transfered = False

        """ FIND TOTAL WORTH AND SEE IF ANY IS ALLOWED TO BE TRANSFERED """
        if (self.denorm(self.total_worth) / self.random_balance) > 1:
            self.transferable = self.norm(self.denorm(self.total_worth) - self.random_balance)
        if (self.denorm(self.total_worth) / self.random_balance) <= 1:
            self.transferable = self.norm(0)

        if chosen_action_amt < 0:  # actions can only be positive
            chosen_action_amt = -chosen_action_amt
        if chosen_action_amt > 1:
            chosen_action_amt = 1
        # print('chosen action = ',chosen_action)
        # print('chosen action amount = ',chosen_action_amt)

        if chosen_action <= len(self.coin_values) * 2:
            for i in range(len(self.coin_values) * 2):  # defining actions for buy/sell coins
                # which coin we're buying/selling
                if chosen_action % 2 == 0:  # if even, we're buying
                    chosen_coin = chosen_action / 2  # 0/2 = 0, 2/2 = 1
                    self.buy_action = True
                else:  # if odd, we're selling
                    chosen_coin = floor(chosen_action / 2)  # 1/2 = 0.5 floor-> 0, 3/2 = 1.5 floor -> 1
                    self.sell_action = True
                chosen_coin -= 1
            chosen_coin = int(chosen_coin)

        """ DENORMALIZED VALUES """
        coin_market_worth_denorm = self.denorm(coin_vals[chosen_coin])
        balance_denorm = self.denorm(self.balance)
        coin_val_denorm = self.coin_amounts[chosen_coin] * coin_market_worth_denorm
        # print(' num coins = ', self.coin_amounts[chosen_coin] )
        # print(' coin val = ', coin_market_worth_denorm)

        buy_amt = (balance_denorm * chosen_action_amt) - self.trans_fee
        sell_amt = (coin_val_denorm * chosen_action_amt)

        # print('coin_val_denorm ', coin_val_denorm)
        if coin_val_denorm > 0 and sell_amt <= (coin_val_denorm + self.trans_fee):
            has_coins = True
        if balance_denorm > self.trans_fee and buy_amt <= (balance_denorm + self.trans_fee) and buy_amt > 0:
            has_money = True
        if chosen_action == len(self.coin_values) * 2 + 1 and self.transferable > 0:
            self.transfer_action = True

        if self.buy_action and has_money:  # Buy (0 mod 2 = 0, 1 mod 2 = 1)
            self.balance = self.norm(balance_denorm - buy_amt)
            self.coin_amounts[chosen_coin] += buy_amt / coin_market_worth_denorm
            # print(' BUYING ' + str(buy_amt))

        elif self.sell_action and has_coins:  # Sell
            self.balance = self.norm(self.denorm(self.balance) + sell_amt)
            # print(sell_amt, coin_vals[chosen_coin])
            self.coin_amounts[chosen_coin] = self.coin_amounts[chosen_coin] - (sell_amt / coin_market_worth_denorm)
            # print(' SELLING ' + str(sell_amt))

        elif self.transfer_action and has_money:  # transfer to bank account
            # print(' TRANSFER ' + str(buy_amt))

            if buy_amt > self.denorm(self.transferable):
                buy_amt = self.denorm(self.transferable)
            self.bank = self.norm(self.bank + (buy_amt + self.trans_fee))  # add money to bank account
            self.balance = self.norm(balance_denorm - buy_amt)
            self.transferable = self.norm(self.denorm(self.transferable) - buy_amt)
            self.money_transfered = True



        else:  # hold
            # print(' HOLDING')
            pass

        self.actions_taken_list.append(chosen_action)

    def create_empty_render_df(self):
        render_df = pd.DataFrame({'dummy_dat': np.zeros((len(self.inputs)))})
        empty_df = render_df.reindex(columns=self.render_keys)

        # for i in range(len(self.render_keys)):
        #
        #     empty_df = render_df.reindex(columns=self.render_keys)#pd.DataFrame({self.render_keys[i]: np.empty((len(self.df_inputs))).tolist()})
        #
        #     if render_df.shape[0] == 0:  # if first time adding a column, use append
        #         render_df = render_df.append(empty_df)
        #
        #     else:  # if not, use concat for correct behaviour
        #         render_df = pd.concat((render_df, empty_df), sort=False, axis=1)

        return empty_df

    def save(self, save_on_step=100):

        self.total_step_counter += 1

        # print('totl steps = ',self.total_step_counter)
        if self.total_step_counter % save_on_step == 0 and self.total_step_counter > 0:
            agent.save_weights('models/ddpg/{}_weights.h5f'.format(ENV_NAME), overwrite=True)


# def make_model_and_stuff():
# Get the environment and extract the number of actions.
env = Stocks(df)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

actor_neurons = 4 * neuron_multiplier
actor_activation = 'relu'
# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(actor_neurons))
actor.add(Activation(actor_activation))
actor.add(Dense(actor_neurons))
actor.add(Activation(actor_activation))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))
print(actor.summary())

critic_neurons = 8 * neuron_multiplier
critic_activation = 'relu'

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(critic_neurons)(x)
x = Activation(critic_activation)(x)
x = Dense(critic_neurons)(x)
x = Activation(critic_activation)(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=500000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0.1, sigma=.3)

agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory,  # nb_steps_warmup_critic=10, nb_steps_warmup_actor=10,
                  random_process=random_process, gamma=.99, target_model_update=1)

agent.compile(Adam(lr=0.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
# agent.load_weights('models/ddpg/{}_weights.h5f'.format(ENV_NAME))
print('Fitting agent.')
agent.fit(env, nb_steps=5000, visualize=True, verbose=1, nb_max_episode_steps=max_training_session_length, )
# agent.fit(env, nb_steps=500000, visualize=False, verbose=1, nb_max_episode_steps=max_training_session_length, )
# agent.fit(env, nb_steps=10000, visualize=True, verbose=1, nb_max_episode_steps=max_training_session_length, )

# After training is done, we save the final weights.
agent.save_weights('models/ddpg/{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# out_df = pd.DataFrame({'agent':[agent]})
# save = open('ddpg_{}_agent.pickle'.format(ENV_NAME), 'wb')
# pickle.dump(out_df, save)
# save.close()
# pickle.dump(agent,'ddpg_{}_agent'.format(ENV_NAME))
# pickle.dump(agent,'ddpg_{}_agent'.format(ENV_NAME))


# Finally, evaluate our algorithm for 5 episodes.
# agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)
