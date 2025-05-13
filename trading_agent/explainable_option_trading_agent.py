import json
import math
import os
import random
import copy
from typing import Literal

from .base_trading_agent import BaseTradingAgent

SUPPORTED_INFERENCE_METHODS = ["forward_predict"]
SUPPORTED_PICKING_ACTION_METHODS = ["random_by_prob", "select_max_prob"]


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
        self.state_history = []
        self.state_p_xt = None

    def load_json(self, path: str) -> dict:
        # Load the JSON file and return the data
        with open(path, "r") as file:
            return json.load(file)

    def reset_state(self) -> None:
        # Reset the state history
        self.state_history = []
        self.state_p_xt = None

    def get_action(self, sentiment_major: str, Date: str, Close: float, **kwargs) -> Literal["buy", "sell"]:
        # Set p(x_t) to if it is None
        if self.state_p_xt is None:
            self.state_p_xt = copy.deepcopy(self.p_xt)
        # Save state history
        self.state_history.append(
            {
                "sentiment_label": sentiment_major,
                "date": Date,
                "price": Close,
                "predicted_xt": None,
                "predicted_action": None,
            }
        )
        if self.inference_method == "forward_predict":
            prob = self.compute_prob_by_forward(**kwargs)

        if self.picking_action_method == "random_by_prob":
            action = random.choices(
                list(prob.keys()),
                weights=list(prob.values()),
                k=1,
            )[0]
            action = "buy" if action == "up" else "sell"
        elif self.picking_action_method == "select_max_prob":
            action = max(prob, key=prob.get)
            action = "buy" if action == "up" else "sell"

        return action

    def compute_prob_by_forward(self, verbose: bool = False, **kwargs) -> dict:
        if verbose:
            print(f"=== get action by forward ===")
        # Predict last x_t if it is None
        if self.state_history[-1]["predicted_xt"] is None:
            if verbose:
                print(f"Predicting x_t for {self.state_history[-1]['date']}")
                print(f"   - Sentiment: {self.state_history[-1]['sentiment_label']}")
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
                    temp = prev_state_p_xt[prev_x] *\
                           self.p_et_given_xt[self.state_history[-1]["sentiment_label"]][next_x] *\
                           self.p_xt_given_xprevt[next_x][prev_x]
                    new_state_p_xt[next_x] += temp
            # Set the new state p_xt and normalize
            self.state_p_xt = new_state_p_xt
            total = sum(self.state_p_xt.values())
            for key in self.state_p_xt.keys():
                self.state_p_xt[key] /= total
            self.state_history[-1]["predicted_xt"] = max(new_state_p_xt, key=new_state_p_xt.get)
            if verbose:
                print(f"New state p_xt: {self.state_p_xt}")
                print(f"Predicted x_t: {self.state_history[-1]['predicted_xt']}")
                print()

        # Predict action
        next_up_prob = self.state_p_xt[self.state_history[-1]["predicted_xt"]] * self.p_xt_given_xprevt["up"][self.state_history[-1]["predicted_xt"]]
        next_down_prob = self.state_p_xt[self.state_history[-1]["predicted_xt"]] * self.p_xt_given_xprevt["down"][self.state_history[-1]["predicted_xt"]]
        total = next_up_prob + next_down_prob

        return_dict = {
            "up": next_up_prob / total,
            "down": next_down_prob / total,
        }
        if verbose:
            print(f"Prob: {return_dict}")
            print(f"=== end get action by forward ===")
        return return_dict
