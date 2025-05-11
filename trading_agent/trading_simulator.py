import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from trading_agent.base_trading_agent import BaseTradingAgent

VALID_BUY_SELL_OPTIONS = ["Open", "Close", "High", "Low"]
VALID_ACTION_OPTIONS = ["buy", "sell"]


@dataclass
class TradingSimulatorConfig:
    price_dataframe: pd.DataFrame | None = None
    price_dataframe_path: str | None = None
    buy_at: Literal["Open", "Close", "High", "Low"] = "Open"
    sell_at: Literal["Open", "Close", "High", "Low"] = "Close"
    trade_type: Literal["portfolio", "option"] = "option"
    reward_factor: float = 1
    loss_factor: float = 1
    initial_balance: float = 100

    def __post_init__(self):
        # Validate the provided price dataframe or path
        if self.price_dataframe is None and self.price_dataframe_path is None:
            raise ValueError("Either price_dataframe or price_dataframe_path must be provided.")
        if self.price_dataframe_path is not None:
            if not os.path.exists(self.price_dataframe_path):
                raise FileNotFoundError(f"The provided path does not exist: {self.price_dataframe_path}")
            self.price_dataframe = pd.read_csv(self.price_dataframe_path)
            if self.price_dataframe.empty:
                raise ValueError("The provided price dataframe is empty.")
        self.price_dataframe["Date"] = pd.to_datetime(self.price_dataframe["Date"])

        # Validate the buy_at and sell_at parameters
        if self.buy_at not in VALID_BUY_SELL_OPTIONS:
            raise ValueError(f"Invalid buy_at option: {self.buy_at}. Must be one of {VALID_BUY_SELL_OPTIONS}.")
        if self.sell_at not in VALID_BUY_SELL_OPTIONS:
            raise ValueError(f"Invalid sell_at option: {self.sell_at}. Must be one of {VALID_BUY_SELL_OPTIONS}.")

        # Validate the trade_type parameter
        if self.trade_type not in ["portfolio", "option"]:
            raise ValueError(f"Invalid trade_type: {self.trade_type}. Must be 'portfolio' or 'option'.")

        # Validate the initial_balance parameter
        if not isinstance(self.initial_balance, (int, float)) or self.initial_balance <= 0:
            raise ValueError("initial_balance must be a positive number.")

        # Validate the loss_factor parameter
        if not isinstance(self.loss_factor, (int, float)) or self.loss_factor <= 0:
            raise ValueError("loss_factor must be a positive number.")

        # Validate the reward_factor parameter
        if not isinstance(self.reward_factor, (int, float)) or self.reward_factor <= 0:
            raise ValueError("reward_factor must be a positive number.")


@dataclass
class Order:
    order_date: pd.Timestamp
    at_price: float
    order_type: Literal["buy", "sell"]
    amount: float
    state: Literal["open", "closed"] = "open"
    reward: float | None = None


run_simulation_result = namedtuple("RunSimulationResult", ["balance", "order_history", "open_orders"])


class TradingSimulator:
    def __init__(self, config: TradingSimulatorConfig):
        self.config = config

    def run_simulation(self, agent: BaseTradingAgent) -> run_simulation_result:
        self.open_orders: list[Order] = []
        self.order_history: list[Order] = []
        self.balance = self.config.initial_balance

        for _, row in self.config.price_dataframe.iterrows():
            current_date = row["Date"]
            current_price = row[self.config.buy_at]
            action = agent.get_action(**row)
            if action not in VALID_ACTION_OPTIONS:
                raise ValueError(f"Invalid action: {action}. Must be 'buy' or 'sell'.")

            new_order = Order(
                order_date=current_date,
                at_price=current_price,
                order_type=action,
                amount=1,  # Assuming a fixed amount for simplicity
            )
            # self.order_history.append(new_order)

            if self.balance <= 0:
                self.balance = 0
                break

            if self.config.trade_type == "option":
                self.option_trade_order_management(new_order, current_price, current_date)
            elif self.config.trade_type == "portfolio":
                self.portfolio_trade_order_management(new_order, current_price, current_date)

        return run_simulation_result(
            balance=self.balance,
            order_history=self.order_history,
            open_orders=self.open_orders,
        )

    def option_trade_order_management(self, new_order: Order, current_price: float, current_date: pd.Timestamp):
        # Process opening orders
        new_open_orders = []

        for open_order in self.open_orders:
            # Check if the order is the date before the current date
            # If so, we can close it
            if open_order.order_date < current_date:
                # Calculate the profit/loss
                reward = 0
                is_price_up = open_order.at_price < current_price
                if open_order.order_type == "buy":
                    if is_price_up:
                        reward = self.config.reward_factor * open_order.amount
                    else:
                        reward = -(self.config.loss_factor * open_order.amount)
                elif open_order.order_type == "sell":
                    if not is_price_up:
                        reward = self.config.reward_factor * open_order.amount
                    else:
                        reward = -(self.config.loss_factor * open_order.amount)
                # Update the balance
                self.balance += reward
                open_order.state = "closed"
                open_order.reward = reward
                self.order_history.append(open_order)
            else:
                new_open_orders.append(open_order)
        self.open_orders = new_open_orders

        # Process the new order
        self.open_orders.append(new_order)

    def portfolio_trade_order_management(self, new_order):
        # Implement portfolio trade order management logic here
        pass
