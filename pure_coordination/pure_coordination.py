import json
import random
import re
from typing import cast

import verifiers as vf
from datasets import Dataset
from verifiers.types import Messages, State

from .config import GameConfig
from .prompt import generate_player_prompt


class PureCoordinationEnv(vf.MultiAgentEnv):
    """Multi-agent environment for a 2-player pure coordination game.

    A simultaneous-move game where:
    - Both players choose an action without seeing the other's choice
    - Both get 1 if they coordinate (same action), 0 otherwise
    - Actions are fixed letter strings, shuffled per player (no focal points)
    """

    # Fixed set of 10 actions (deterministic, no randomness)
    FIXED_ACTIONS = [
        "qahf",
        "trxc",
        "kafn",
        "afqo",
        "fpva",
        "usie",
        "yicc",
        "wpus",
        "nzjo",
        "vqwp",
    ]

    def __init__(
        self,
        num_train_examples: int = 1000,
        num_eval_examples: int = 100,
        num_actions: int = 10,
        **kwargs,
    ):
        self.config = GameConfig(num_actions=num_actions)

        self.num_train_examples = num_train_examples
        self.num_eval_examples = num_eval_examples

        # Build datasets - each example is just a seed
        train_dataset = Dataset.from_list(
            [{"question": self._get_initial_observation(seed=i), "answer": str(i)} for i in range(num_train_examples)]
        )

        eval_dataset = Dataset.from_list(
            [
                {"question": self._get_initial_observation(seed=i), "answer": str(i)}
                for i in range(num_train_examples, num_train_examples + num_eval_examples)
            ]
        )

        # Two players
        agent_ids = ["player_1", "player_2"]

        super().__init__(
            dataset=train_dataset,
            eval_dataset=eval_dataset,
            max_turns=2,
            protocol=vf.RoundRobinProtocol(agent_ids),
            **kwargs,
        )

        # Register agents
        self.register_agent(
            vf.Agent(
                id="player_1",
                system_prompt=generate_player_prompt(self.config, "player_1"),
                is_trainable=True,
            )
        )
        self.register_agent(
            vf.Agent(
                id="player_2",
                system_prompt=generate_player_prompt(self.config, "player_2"),
                is_trainable=True,
            )
        )

    def _generate_actions(self) -> list[str]:
        """Return the fixed action labels for a game (unshuffled base set)."""
        return self.FIXED_ACTIONS[: self.config.num_actions]

    def _shuffle_for_player(self, actions: list[str], seed: int, player_id: str) -> list[str]:
        """Shuffle actions differently for each player to remove positional focal points."""
        player_seed = seed + (1 if player_id == "player_1" else 2)
        rng = random.Random(player_seed)
        shuffled = list(actions)
        rng.shuffle(shuffled)
        return shuffled

    def _get_initial_observation(self, seed: int) -> str:
        """Generate the initial game observation for a given seed."""
        actions = self._generate_actions()
        action_list = ", ".join(actions)
        return f"Coordination game. Choose one action from: {action_list}"

    def _initialize_game(self, seed: int) -> State:
        """Create a fresh game state from a seed."""
        actions = self._generate_actions()
        return cast(
            State,
            {
                "seed": seed,
                "actions": actions,
                "player_1_action": None,
                "player_2_action": None,
                "is_complete": False,
                "coordinated": False,
                "payoff": 0.0,
            },
        )

    def _get_player_observation(self, state: State, player_id: str) -> str:
        """Generate observation for a player (shuffled differently per player)."""
        actions = self._shuffle_for_player(state["actions"], state["seed"], player_id)
        action_list = ", ".join(actions)
        return f"Choose your action. Available actions: {action_list}"

    def _get_final_observation(self, state: State) -> str:
        """Generate final observation with results."""
        result = {
            "game_complete": True,
            "coordinated": state.get("coordinated", False),
            "player_1_action": state.get("player_1_action"),
            "player_2_action": state.get("player_2_action"),
            "payoff": state.get("payoff", 0.0),
        }
        return f"Game Complete!\n\n{json.dumps(result, indent=2)}"

    async def setup_state(self, state: State) -> State:
        """Initialize game state."""
        state = await super().setup_state(state)
        seed = int(state["answer"])
        state.update(self._initialize_game(seed))
        state["agent_messages"] = {}
        return state

    @vf.stop
    async def game_complete(self, state: State) -> bool:
        return state.get("is_complete", False)

    async def build_agent_prompt(self, agent_id: str, state: State) -> Messages:
        """Build prompt for the given agent."""
        agent = self.get_agent(agent_id)

        if agent_id not in state["agent_messages"]:
            state["agent_messages"][agent_id] = [{"role": "system", "content": agent.system_prompt}]

        messages = state["agent_messages"][agent_id]
        observation = self._get_player_observation(state, agent_id)
        messages.append({"role": "user", "content": observation})
        return list(messages)

    async def on_turn_complete(self, state: State) -> None:
        """Process the action after model response."""
        if not state["trajectory"]:
            return

        last_step = state["trajectory"][-1]
        completion = last_step.get("completion", [])
        if not completion:
            return

        content = ""
        for msg in completion:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                break

        agent_id = state["extras"].get("current_agent_id")
        if agent_id and agent_id in state["agent_messages"]:
            state["agent_messages"][agent_id].append({"role": "assistant", "content": content})

        if agent_id == "player_1":
            self._process_player_turn(state, content, "player_1")
        elif agent_id == "player_2":
            self._process_player_turn(state, content, "player_2")
            self._finalize_game(state)

    def _process_player_turn(self, state: State, content: str, player_id: str) -> None:
        """Process a player's action choice."""
        actions = state["actions"]
        action = None
        match = re.search(r"<answer>\s*([a-z]+)\s*</answer>", content, re.IGNORECASE)
        if match:
            choice = match.group(1).lower()
            if choice in actions:
                action = choice

        if action is None:
            # Invalid format - use random action
            rng = random.Random(state["seed"] + (1000 if player_id == "player_1" else 2000))
            action = rng.choice(actions)

        state[f"{player_id}_action"] = action

    def _finalize_game(self, state: State) -> None:
        """Compute payoffs and finalize the game."""
        p1_action = state["player_1_action"]
        p2_action = state["player_2_action"]

        state["coordinated"] = p1_action == p2_action
        state["payoff"] = self.config.get_payoff(p1_action, p2_action)
        state["is_complete"] = True

        # Set final response
        state["final_env_response"] = [{"role": "user", "content": self._get_final_observation(state)}]


def coordination_reward_func(parser, completion: Messages, **_kwargs) -> float:
    """Reward: 1 if coordinated, 0 otherwise."""
    for msg in reversed(parser.get_user_messages(completion)):
        content = msg.get("content", "")
        if isinstance(content, str) and "game_complete" in content.lower():
            try:
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    data = json.loads(content[json_start:json_end])
                    return data.get("payoff", 0.0)
            except (json.JSONDecodeError, ValueError):
                continue
    return 0.0


def load_environment(
    num_train_examples: int = 1000,
    num_eval_examples: int = 100,
    num_actions: int = 10,
) -> vf.Environment:
    """Load the Coordination Game environment.

    Args:
        num_train_examples: Number of training examples
        num_eval_examples: Number of evaluation examples
        num_actions: Number of available actions (default: 10)

    Returns:
        Configured PureCoordinationEnv instance
    """
    parser = vf.XMLParser(fields=["answer"], answer_field="answer")
    rubric = vf.Rubric(parser=parser)

    rubric.add_reward_func(coordination_reward_func, weight=0.9)

    format_reward = parser.get_format_reward_func()
    format_reward.__name__ = "format_reward"
    rubric.add_reward_func(format_reward, weight=0.1)

    return PureCoordinationEnv(
        parser=parser,
        rubric=rubric,
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        num_actions=num_actions,
    )
