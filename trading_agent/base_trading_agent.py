from abc import ABC, abstractmethod
from typing import Literal


class BaseTradingAgent(ABC):
    @abstractmethod
    def get_action(self, **kwargs) -> Literal["buy", "sell", "hold"]:
        """
        Given the current state of the environment, return an action.
        :param state: The current state of the environment.
        :return: An action to be taken in the environment.
        """
        pass

    def reset_state(self):
        """
        Reset the state of the agent.
        This method is called at the beginning of each episode.
        """
        pass
