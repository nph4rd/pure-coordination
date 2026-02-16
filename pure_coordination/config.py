from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    """Configuration for a coordination game.

    Parameters:
        num_actions: Number of available actions for each player
        coordinate_payoff: Payoff when both players choose the same action
        miscoordinate_payoff: Payoff when players choose different actions
    """

    num_actions: int = 2
    coordinate_payoff: float = 1.0
    miscoordinate_payoff: float = 0.0

    def __post_init__(self) -> None:
        if self.num_actions < 2:
            raise ValueError("num_actions must be at least 2")

    def get_payoff(self, p1_action: str, p2_action: str) -> float:
        """Get payoff for a given action pair (same for both players)."""
        if p1_action == p2_action:
            return self.coordinate_payoff
        return self.miscoordinate_payoff
