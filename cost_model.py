"""Cost per token from measured throughput and an hourly hardware price.

cost per token = hourly price / (tokens per second * 3600)

The hourly price is an input. It is not measured here, so absolute dollar
figures only hold for the price you supply. Comparing approaches at the same
price is a direct comparison of their throughput.
"""


def cost_per_token(throughput_tok_s: float, hourly_usd: float) -> float:
    """USD per generated token at a given throughput and hourly hardware price."""
    if throughput_tok_s <= 0:
        raise ValueError("throughput_tok_s must be positive")
    return hourly_usd / (throughput_tok_s * 3600.0)


def cost_per_million_tokens(throughput_tok_s: float, hourly_usd: float) -> float:
    """USD per one million generated tokens."""
    return cost_per_token(throughput_tok_s, hourly_usd) * 1e6
