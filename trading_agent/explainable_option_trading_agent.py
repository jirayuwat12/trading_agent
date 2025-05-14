import copy
import json
import os
import random
from dataclasses import dataclass
from typing import Literal

from .base_trading_agent import BaseTradingAgent

SUPPORTED_INFERENCE_METHODS = ["forward_predict"]
SUPPORTED_PICKING_ACTION_METHODS = ["random_by_prob", "select_max_prob"]


@dataclass
class State:
    sentiment_label: str = ""
    date: str = ""
    price: str = ""
    predicted_xt: Literal["up", "down"] | None = None
    predicted_action: Literal["buy", "sell"] | None = None


class ExplainableOptionTradingAgent(BaseTradingAgent):
    def __init__(
        self,
        p_et_path: str,
        p_xt_path: str,
        p_et_given_xt_path: str,
        p_xt_given_xprevt_path: str,
        inference_method: Literal["forward_predict"] = "forward_predict",
        picking_action_method: Literal["random_by_prob", "select_max_prob"] = "select_max_prob",
    ) -> None:
        """
        Explainable Option Trading Agent which uses a probabilistic model to predict the next action based on the sentiment label and previous state.

        :param p_et_path: Path to the JSON file containing P(e_t)
        :param p_xt_path: Path to the JSON file containing P(x_t)
        :param p_et_given_xt_path: Path to the JSON file containing P(e_t | x_t)
        :param p_xt_given_xprevt_path: Path to the JSON file containing P(x_t | x_t-1)
        :param inference_method: Method to use for inference. Currently only "forward_predict" is supported.
        :param picking_action_method: Method to use for picking the action. Options are "random_by_prob" or "select_max_prob".

        :raises FileNotFoundError: If any of the provided paths do not exist.
        :raises ValueError: If the inference method or picking action method is not supported.
        """
        for given_path in [p_et_path, p_xt_path, p_et_given_xt_path, p_xt_given_xprevt_path]:
            if not os.path.exists(given_path):
                raise FileNotFoundError(f"Path does not exist: {given_path}")

        # Load the JSON files
        self.p_et = self.load_json(p_et_path)
        self.p_xt = self.load_json(p_xt_path)
        self.p_et_given_xt = self.load_json(p_et_given_xt_path)
        self.p_xt_given_xprevt = self.load_json(p_xt_given_xprevt_path)

        # Validate the inference method
        if inference_method not in SUPPORTED_INFERENCE_METHODS:
            raise ValueError(
                f"Invalid inference method: {inference_method}. Supported methods are: {SUPPORTED_INFERENCE_METHODS}"
            )
        self.inference_method = inference_method

        # Validate the picking action method
        if picking_action_method not in SUPPORTED_PICKING_ACTION_METHODS:
            raise ValueError(
                f"Invalid picking action method: {picking_action_method}. Supported methods are: {SUPPORTED_PICKING_ACTION_METHODS}"
            )
        self.picking_action_method = picking_action_method

        # State management
        self.state_history: list[State] = []
        self.state_p_xt = None

    def load_json(self, path: str) -> dict:
        """
        Load a JSON file and return the data.

        :param path: Path to the JSON file.

        :return: Data loaded from the JSON file.

        :raises FileNotFoundError: If the file does not exist.
        """
        # Check if the file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")
        # Load the JSON file and return the data
        with open(path, "r") as file:
            return json.load(file)

    def reset_state(self) -> None:
        """
        Reset the state of the agent. This is useful for starting a new trading session or when the agent needs to be reinitialized.
        This method clears the state history and resets the state probability.
        """
        # Reset the state history
        self.state_history: list[State] = []
        self.state_p_xt = None

    def get_action(self, sentiment_major: str, Date: str, Close: float, **kwargs) -> Literal["buy", "sell"]:
        """
        Get the action to take based on the sentiment label and previous state.
        The action is determined by the probabilistic model and the specified picking action method.

        :param sentiment_major: The sentiment label (e.g., "positive", "negative").
        :param Date: The date of the current state.
        :param Close: The closing price of the asset.
        :param kwargs: Additional keyword arguments for future use.

        :return: The action to take ("buy" or "sell").
        """
        # Set p(x_t) to if it is None
        if self.state_p_xt is None:
            self.state_p_xt = copy.deepcopy(self.p_xt)
        # Save state history
        self.state_history.append(
            State(
                sentiment_label=sentiment_major,
                date=Date,
                price=Close,
                predicted_xt=None,
                predicted_action=None,
            )
        )
        if self.inference_method == "forward_predict":
            prob = self.compute_prob_by_forward(**kwargs)

        if self.picking_action_method == "random_by_prob":
            next_xt = random.choices(
                list(prob.keys()),
                weights=list(prob.values()),
                k=1,
            )[0]
        elif self.picking_action_method == "select_max_prob":
            next_xt = max(prob, key=prob.get)

        action = "buy" if next_xt == "up" else "sell"

        return action

    def compute_prob_by_forward(self, verbose: bool = False, **kwargs) -> dict:
        """
        Compute the probability of the next action using the forward prediction method.
        This method uses the current state and the probabilistic model to predict the next action.

        :param verbose: If True, print detailed information about the computation.
        :param kwargs: Additional keyword arguments for future use.

        :return: A dictionary containing the probabilities of the next action ("up" and "down").
        """
        if verbose:
            print(f"=== get action by forward ===")
        # Predict last x_t if it is None
        if self.state_history[-1].predicted_xt is None:
            if verbose:
                print(f"Predicting x_t for {self.state_history[-1].date}")
                print(f"   - Sentiment: {self.state_history[-1].sentiment_label}")
            # Filtering: predict x_t from e_1:e_t
            prev_state_p_xt = copy.deepcopy(self.state_p_xt)
            if verbose:
                print(f"Prev state p_xt: {self.state_p_xt}")
            new_state_p_xt = {}
            for next_x in self.p_xt.keys():
                # P(x_t = next_x) = P(x_t-1 =   up) * P(e_t = last_sentiment | x_t = next_x) * P(x_t = next_x | x_t-1 = up) +
                #                   P(x_t-1 = down) * P(e_t = last_sentiment | x_t = next_x) * P(x_t = next_x | x_t-1 = down)
                # For next_t = {up, down}
                new_state_p_xt[next_x] = 0
                for prev_x in self.p_xt.keys():
                    temp = (
                        prev_state_p_xt[prev_x]
                        * self.p_et_given_xt[self.state_history[-1].sentiment_label][next_x]
                        * self.p_xt_given_xprevt[next_x][prev_x]
                    )
                    new_state_p_xt[next_x] += temp
            # Set the new state p_xt and normalize
            self.state_p_xt = new_state_p_xt
            total = sum(self.state_p_xt.values())
            for key in self.state_p_xt.keys():
                self.state_p_xt[key] /= total
            self.state_history[-1].predicted_xt = max(self.state_p_xt, key=self.state_p_xt.get)
            if verbose:
                print(f"New state p_xt: {self.state_p_xt}")
                print(f"Predicted x_t: {self.state_history[-1].predicted_xt}")
                print()

        # Predict action
        next_up_prob = (
            self.state_p_xt[self.state_history[-1].predicted_xt]
            * self.p_xt_given_xprevt["up"][self.state_history[-1].predicted_xt]
        )
        next_down_prob = (
            self.state_p_xt[self.state_history[-1].predicted_xt]
            * self.p_xt_given_xprevt["down"][self.state_history[-1].predicted_xt]
        )
        total = next_up_prob + next_down_prob

        return_dict = {
            "up": next_up_prob / total,
            "down": next_down_prob / total,
        }
        if verbose:
            print(f"Prob: {return_dict}")
            print(f"=== end get action by forward ===")
        return return_dict
