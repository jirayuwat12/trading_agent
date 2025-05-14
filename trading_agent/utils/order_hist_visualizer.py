import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..trading_simulator import Order


def plot_order_history(
    orders: list[Order],
    initial_balance: float,
    title: str = "Order History",
    test_split_date: str = None,
    marker_size: int = 70,
    marker_alpha: float = 0.5,
    line_alpha: float = 0.3,
) -> None:
    """
    Plots the order history of a trading agent.

    :param orders: List of Order objects containing order history.
    :param initial_balance: Initial balance of the trading agent.
    :param title: Title of the plot.
    :param test_split_date: Date to split the training and testing data.
    :return: None
    """
    win_order: list[Order] = []
    lose_order: list[Order] = []
    prices: list[float] = []
    dates: list[str] = []

    # Extracting prices and dates from orders
    for order in orders:
        prices.append(order.at_price)
        dates.append(order.order_date)

    # Extracting win and lose orders
    for order in orders:
        if order.state == "closed":
            if order.reward > 0:
                win_order.append(order)
            else:
                lose_order.append(order)

    # Plotting
    plt.figure(figsize=(15, 10))

    # Plot the price history
    plt.subplot(2, 1, 1)
    sns.lineplot(x=dates, y=prices, label="Price", color="blue", alpha=line_alpha)

    # Plot the win and lose orders
    for order_type, marker in [("buy", "^"), ("sell", "v")]:
        sns.scatterplot(
            x=[order.order_date for order in win_order if order.order_type == order_type],
            y=[order.at_price for order in win_order if order.order_type == order_type],
            color="green",
            label=f"Win Orders ({order_type.capitalize()})",
            marker=marker,
            s=marker_size,
            alpha=marker_alpha,
        )
        sns.scatterplot(
            x=[order.order_date for order in lose_order if order.order_type == order_type],
            y=[order.at_price for order in lose_order if order.order_type == order_type],
            color="red",
            label=f"Lose Orders ({order_type.capitalize()})",
            marker=marker,
            s=marker_size,
            alpha=marker_alpha,
        )

    # If test_split_date is provided, add a vertical line
    if test_split_date:
        # plt.axvline(test_split_date, color="orange", linestyle="--", label="Test Split Date")
        plt.axvline(
            pd.to_datetime(test_split_date),
            color="orange",
            linestyle="--",
            label="Test Split Date",
        )
        plt.text(
            pd.to_datetime(test_split_date),
            max(prices),
            "Test Split Date",
            color="orange",
            fontsize=10,
            ha="left",
        )

    # Add labels and title
    sns.set_style("whitegrid")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)

    # Plot balance history
    plt.subplot(2, 1, 2)
    rewards = [order.reward if order.reward is not None else 0 for order in orders]
    cumulative_rewards = [sum(rewards[: i + 1]) / initial_balance for i in range(len(rewards))]
    cumulative_rewards = np.array(cumulative_rewards) * 100

    sns.lineplot(
        x=dates,
        y=cumulative_rewards,
        label="Cumulative Rewards",
        color="purple",
        alpha=0.5,
    )
    plt.title("Cumulative Rewards (percentage of Initial Balance)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Reward (%)")
    plt.xticks(rotation=45)

    # Add vertical line for test split date
    if test_split_date:
        plt.axvline(
            pd.to_datetime(test_split_date),
            color="orange",
            linestyle="--",
            label="Test Split Date",
        )
        plt.text(
            pd.to_datetime(test_split_date),
            max(cumulative_rewards),
            "Test Split Date",
            color="orange",
            fontsize=10,
            ha="left",
        )

    # Add 0% line
    plt.axhline(0, color="black", linestyle="--", label="0% Line")
    plt.fill_between(
        dates,
        0,
        cumulative_rewards,
        where=(cumulative_rewards > 0),
        color="green",
        alpha=0.3,
    )
    plt.fill_between(
        dates,
        0,
        cumulative_rewards,
        where=(cumulative_rewards < 0),
        color="red",
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()
    plt.show()
