# Pure-Coordination

### Overview
- **Environment ID**: `pure-coordination`
- **Short description**: 2-player coordination game where players must simultaneously choose the same action
- **Tags**: multi-agent, single-turn, cooperative

### Task
- **Type**: single-turn
- **Parser**: XMLParser (fields: answer)
- **Rubric**: Score-based reward (0-1)

### Description

A pure coordination game is a game-theoretic scenario where two players must simultaneously choose an action without seeing each other's choice. Both players receive a payoff of 1.0 if they coordinate (choose the same action), or 0.0 if they miscoordinate.

  - Players: 2
  - Actions: Random 4-character letter strings with no inherent meaning
  - Perfect score: 1.0 (successful coordination)

The key challenge is that actions have no focal points or prior meaning—they are randomly generated strings. Additionally, the action list is shuffled differently for each player, preventing positional coordination strategies. This tests pure coordination ability without relying on arbitrary conventions.

### Dependencies
- `verifiers>=0.1.8`

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval pure-coordination
```

Configure model and sampling:

```bash
uv run vf-eval pure-coordination -m gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7 -a '{"num_actions": 3}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `1000` | Number of training examples (each with a unique seed) |
| `num_eval_examples` | int | `100` | Number of evaluation examples |
| `num_actions` | int | `2` | Number of available actions per game |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Coordination payoff (1.0 if both players chose the same action, 0.0 otherwise) |

### Project Structure

```
pure_coordination/
├── config.py              # GameConfig dataclass with payoff settings
├── prompt.py              # System prompt template for players
└── pure_coordination.py   # PureCoordinationEnv environment and reward function
```
