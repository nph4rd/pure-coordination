from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import GameConfig


def generate_player_prompt(config: "GameConfig", player_id: str) -> str:
    """Generate system prompt for a player."""
    return f"""You are {player_id.upper()} in a coordination game.

## THE GAME

You and another player must each choose an action SIMULTANEOUSLY.
Neither of you can see the other's choice before making your own.

## ACTIONS

Each round, you will be given a list of {config.num_actions} available actions.
The actions are random letter strings with no inherent meaning or order.

## PAYOFFS

- If you BOTH choose the SAME action: you each get {config.coordinate_payoff} point(s)
- If you choose DIFFERENT actions: you each get {config.miscoordinate_payoff} point(s)

## YOUR GOAL

Coordinate with the other player to choose the same action.

## OUTPUT FORMAT

Respond with ONLY your chosen action in <answer></answer> tags. Nothing else.
"""
