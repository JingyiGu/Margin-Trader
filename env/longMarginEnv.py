from __future__ import annotations

from typing import List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")


from stable_baselines3.common.logger import Logger, KVWriter, CSVOutputFormat


class LongMarginTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            df: pd.DataFrame,
            stock_dim: int,
            hmax: int,
            initial_amount: int,
            num_stock_shares: list[int],
            buy_cost_pct: list[float],
            sell_cost_pct: list[float],
            reward_scaling: float,
            state_space: int,
            action_space: int,
            tech_indicator_list: list[str],
            turbulence_threshold=None,
            risk_indicator_col="turbulence",
            make_plots: bool = False,
            print_verbosity=10,
            margin=2,  # can borrow 100% cash from broker
            long_short_ratio=1,  # deposit for long:short = 1:1
            maintenance = 0.4,
            penalty_sharpe = 0.001,
            day=0,
            initial=True,
            previous_state=[],
            model_name="",
            mode="",
            iteration="",
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.margin = margin  # 2
        self.long_short_ratio = long_short_ratio  # 1
        self.maintenance = maintenance
        self.penalty_sharpe = penalty_sharpe
        # initalize state
        # self.state = self._initiate_state()
        self.state = self._initiate_margin_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1: 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = (
            []
        )  # we need sometimes to preserve the state in the middle of trading process
        self.date_memory = [self._get_date()]
        self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()
        self._seed()

    def _sell_long_stock(self, index, action):
        def _do_sell_long_normal():
            if (
                    self.state[index + 2 * self.stock_dim + 3] != True
            ):  # check if the stock is able to sell, for simlicity we just add it in techical index
                # if self.state[index + 1] > 0: # if we use price<0 to denote a stock is unable to trade in that day, the total asset calculation may be wrong for the price is unreasonable
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 3] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 3]
                    )
                    sell_amount = (
                            self.state[index + 3]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                    )
                    # update balance
                    # print('sell amount:', sell_amount)
                    self.state[0] += sell_amount  # cash
                    self.state[2] -= self.state[index + 3]  * sell_num_shares  * self.sell_cost_pct[index]
                    self.state[index + self.stock_dim + 3] -= sell_num_shares  # holding shares
                    self.cost += (
                            self.state[index + 3]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
                    # print('sell amount:', 0)
            else:
                sell_num_shares = 0
                # print('sell amount:', 0)

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 3] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 3] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 3]
                        sell_amount = (
                                self.state[index + 3]
                                * sell_num_shares
                                * (1 - self.sell_cost_pct[index])
                        )
                        # print('sell amount:', sell_amount)
                        # update balance
                        self.state[0] += sell_amount
                        self.state[2] -= self.state[index + 3]  * sell_num_shares  * self.sell_cost_pct[index]
                        self.state[index + self.stock_dim + 3] = 0
                        self.cost += (
                                self.state[index + 3]
                                * sell_num_shares
                                * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                        # print('sell amount:', 0)
                else:
                    sell_num_shares = 0
                    # print('sell amount:', 0)
            else:
                sell_num_shares = _do_sell_long_normal()
        else:
            sell_num_shares = _do_sell_long_normal()

        return sell_num_shares

    def _buy_long_stock(self, index, action):
        def _do_long_buy():
            if (
                    self.state[index + 2 * self.stock_dim + 1] != True and self._check_long_maintenance()>self.maintenance
                    # check if equity /long market > 0.4, if not satisfied go to else statement
            ):  # check if the stock is able to buy
                # if self.state[index + 1] >0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_shares = self.state[0] // (
                        self.state[index + 3] * (1 + self.buy_cost_pct[index])
                )  # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
                # print('available_amount:{}'.format(available_amount))

                # update balance
                buy_num_shares = min(available_shares, action)
                buy_amount = (
                        self.state[index + 3]
                        * buy_num_shares
                        * (1 + self.buy_cost_pct[index])
                )
                # print('buy amount:', buy_amount)
                self.state[0] -= buy_amount
                self.state[2] -= self.state[index + 3] * buy_num_shares * self.buy_cost_pct[index]
                self.state[index + self.stock_dim + 3] += buy_num_shares
                self.cost += (
                        self.state[index + 3] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0
                # print('buy amount:', 0)               

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_long_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_long_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares


    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def step(self, actions):
    
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[2] 
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = end_total_asset - self.asset_memory[0] # initial_amount is only cash part of our initial asset
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                        (252 ** 0.5)
                        * df_total_value["daily_return"].mean()
                        / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                plt.close()

            # Add outputs to logger interface
            self.logger.record("environment/portfolio_value", end_total_asset)
            self.logger.record("environment/total_reward", tot_reward)
            self.logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            self.logger.record("environment/total_cost", self.cost)
            self.logger.record("environment/total_trades", self.trades)

            return self.state, self.reward, self.terminal, False, {}

        else:
            # print('################ day: {} ################ '.format(self.day))
            # print('State:', self.state[:3 + self.stock_dim * 3])
            actions = actions * self.hmax  # actions initially is scaled between -1 to 1
            long_actions = actions.astype(int)  # convert into integer because we can't by fraction of shares
            # print('Actions from agent:', actions)
            # print('Actions after combine long and short positions:', actions)
            
            begin_total_asset = self.state[2]  
            # print("begin_total_asset:{}".format(begin_total_asset))

            ################ long position
            # print('############## long position')
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    long_actions = np.array([-self.hmax] * self.stock_dim)

            argsort_long_actions = np.argsort(long_actions)
            sell_long_index = argsort_long_actions[: np.where(long_actions < 0)[0].shape[0]]
            buy_long_index = argsort_long_actions[::-1][: np.where(long_actions > 0)[0].shape[0]]

            # print('selling!')
            for index in sell_long_index:
                # print('stock index:', index)
                long_actions[index] = self._sell_long_stock(index, long_actions[index]) * (-1)

            # print('buying!')
            for index in buy_long_index:
                # print('stock index:', index)
                long_actions[index] = self._buy_long_stock(index, long_actions[index])
                # print('buy shares: ', long_actions[index])
                # print(self.state)

            self.actions_memory.append(long_actions)
            # check margin requirement every 30 days and update the equity, market and loan respectively
            

            if self.day != 0 and self.day % 30 == 0:
                self._update_loan()
            else:
                if self._check_long_maintenance() < 0.3:
                    self._update_loan()

            ############## state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day, :]

            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()
            end_total_asset = self.state[2]
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = end_total_asset - begin_total_asset

            if len(self.asset_memory)>=5:
                daily_return = pd.Series(self.asset_memory[-5:]).pct_change(1)
                sharpe = (
                        (252**0.5)
                        * daily_return.mean()
                        / daily_return.std()
                    )
            else:
                sharpe = 0

            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling + sharpe * self.penalty_sharpe
            self.state_memory.append(
                self.state
            )
            
            # print("end_total_asset:{}".format(end_total_asset))

            # print('reward:', self.reward)
            

        return self.state, self.reward, self.terminal, False, {}

    def reset(
            self,
            *,
            seed=None,
            options=None,
    ):
        # initiate state
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_margin_state()

        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[3: 3 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[3: (self.stock_dim + 3)])
                * np.array(
                    self.previous_state[(self.stock_dim + 3): (self.stock_dim * 2 + 3)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state

    def _check_long_maintenance(self):
        market = np.array(self.state[3: (self.stock_dim + 3)]) * np.array(
            self.state[(self.stock_dim + 3): (self.stock_dim * 2 + 3)])
        long_market = sum(market[market > 0])
        equity = self.state[2]

        maintenance = equity/long_market if long_market != 0 else 1
        
        return maintenance

    def _initiate_margin_state(self):
        # equity_long = self.long_short_ratio / (self.long_short_ratio + 1) * self.initial_amount  # 50k
        equity_long = self.initial_amount
        available_cash_long = equity_long * self.margin  # 100k
        loan = equity_long * (self.margin - 1)  # 50k
        state = (
                [available_cash_long, loan, equity_long]  # cash, loan, equity
                + self.data.close.values.tolist()
                + self.num_stock_shares
                + sum(
            (
                self.data[tech].values.tolist()
                for tech in self.tech_indicator_list
            ),
            [],
        )
        )
        return state


    def _update_loan(self):
        long_cash, loan = self.state[0], self.state[1]
        market = np.array(self.state[3: (self.stock_dim + 3)]) * np.array(
            self.state[(self.stock_dim + 3): (self.stock_dim * 2 + 3)])
        long_market = sum(market[market > 0])
        long_equity = self.state[2]
        loan_diff = long_equity - loan
        
        if loan_diff > 0:  # earn profit, can borrow more cash
            loan = long_equity
            long_cash += loan_diff
            self.state[0:2] = [long_cash, loan]
        else:  # loss, need to return some cash to loan. if still not meet requirement, sell long
            # return_cash = min(long_cash, abs(loan_diff))
            
            if long_cash > abs(loan_diff):
                long_cash -= abs(loan_diff)
                loan -= abs(loan_diff)
                self.state[0:2] = [long_cash, loan]
            else:
                loan_rest = abs(loan_diff) - long_cash  # still owe this value
                long_cash = 0  # first return all long_cash
                self.state[0] = 0              
                argsort_market = np.argsort(market)  # sort based on market value, index with small value first

                for i in argsort_market:
                    if market[i] > 0:
                        self._sell_long_stock(i, self.state[self.stock_dim + 3 + i])  # sell the one with the lowest market value, change the market and cash in state in this process
                        if market[i] < loan_rest: 
                            loan_rest -= market[i]
                            self.state[0] = 0 # return cash to loan
                        else:
                            self.state[0] -= loan_rest # return cash to loan, and keep rest of cash
                            break  # if satisfied stop
                self.state[1] -= abs(loan_diff)
          
                
    def _update_state(self):

        if len(self.df.tic.unique()) > 1:

            long_cash, loan = self.state[0], self.state[1]
            market = np.array(self.data.close.values.tolist()) * np.array(
                self.state[(self.stock_dim + 3): (self.stock_dim * 2 + 3)])
            long_market = sum(market[market > 0])
            long_equity = long_cash + long_market - loan
            
            # for multiple stock
            state = (
                    [long_cash, loan, long_equity]  #
                    + self.data.close.values.tolist()
                    + list(self.state[(self.stock_dim + 3): (self.stock_dim * 2 + 3)])
                    + sum(
                (
                    self.data[tech].values.tolist()
                    for tech in self.tech_indicator_list
                ),
                [],
            )
            )
        else:
            # for single stock
            state = (
                    [self.state[0]]
                    + [self.data.close]
                    + list(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = [x[:3+2*self.stock_dim] for x in self.state_memory]
            df_states = pd.DataFrame(
                state_list,
                columns=['cash','loan','long_equity']+[x+"_c" for x in self.data.tic.values]+[x+"_h" for x in self.data.tic.values],
            )
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        return df_states


    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = [x + "_l" for x in self.data.tic.values] 
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
