import json
import math
import os
import random
from typing import Literal

from .base_trading_agent import BaseTradingAgent

SUPPORTED_INFERENCE_METHODS = ["viterbi"]
SUPPORTED_PICKING_ACTION_METHODS = ["random_by_prob", "select_max_prob"]


class ExplainableOptionTradingAgent(BaseTradingAgent):
    def __init__(
        self,
        p_et_path: str,
        p_xt_path: str,
        p_et_given_xt_path: str,
        p_xt_given_xprevt_path: str,
        inference_method: Literal["viterbi"] = "viterbi",
        viterbi_evidence_num: int = 1,
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

        # Viterbi configuration
        self.viterbi_evidence_num = viterbi_evidence_num

        # State management
        self.state_history = []

    def load_json(self, path: str) -> dict:
        # Load the JSON file and return the data
        with open(path, "r") as file:
            return json.load(file)

    def get_action_by_viterbi(self, **kwargs) -> Literal["buy", "sell"]:
        current_p_xt = self.p_xt.copy()
        for xt in current_p_xt:
            current_p_xt[xt] = math.log(current_p_xt[xt])

        for i in range(max(0, len(self.state_history) - self.viterbi_evidence_num), len(self.state_history)):
            current_et = self.state_history[i]["sentiment_label"]
            for xt in current_p_xt:
                if i != 0:
                    prev_xt = self.state_history[i - 1]["x_t"]
                    current_p_xt[xt] += math.log(self.p_xt_given_xprevt[xt][prev_xt])
                current_p_xt[xt] += math.log(self.p_et_given_xt[current_et][xt])
            # Set the prob the the max prob of the current state
            # max_prob = max(current_p_xt.values())
            # for xt in current_p_xt:
            #     current_p_xt[xt] = max_prob

        if self.picking_action_method == "select_max_prob":
            action = "buy" if current_p_xt["up"] > current_p_xt["down"] else "sell"
        elif self.picking_action_method == "random_by_prob":
            action = random.choices(
                ["buy", "sell"],
                # weights=[current_p_xt["up"], current_p_xt["down"]],
                weights=[math.exp(current_p_xt["up"]), math.exp(current_p_xt["down"])],
                k=1,
            )[0]

        return action

    def get_action(self, sentiment_label: str, Date: str, Close: float, x_t: str, **kwargs) -> Literal["buy", "sell"]:
        # Save state history
        self.state_history.append(
            {
                "sentiment_label": sentiment_label,
                "date": Date,
                "price": Close,
                "x_t": "up" if x_t == 1 else "down",
            }
        )
        if self.inference_method == "viterbi":
            action = self.get_action_by_viterbi(**kwargs)

        return action
