import random

from .base_trading_agent import BaseTradingAgent


class RandomTradingAgent(BaseTradingAgent):
    def __init__(self, possible_actions: list[str] = ["buy", "sell"]):
        """
        Random Trading Agent which randomly chooses an action from the list of possible actions.

        :param possible_actions: List of possible actions. Default is ["buy", "sell"].
        """
        self.possible_actions = possible_actions

    def get_action(self, **kwargs) -> str:
        """
        Randomly chooses an action from the list of possible actions.
        """
        # Randomly choose an action: buy, sell, or hold
        return random.choice(self.possible_actions)
