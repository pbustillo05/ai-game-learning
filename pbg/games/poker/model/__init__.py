"""Poker neural-network components, exposed lazily to avoid import cycles."""

__all__ = [
    "PokerModelConfig",
    "PokerPublicBatch",
    "PolicyOutput",
    "PostflopMasks",
    "PostflopNet",
    "PreflopNet",
]


def __getattr__(name: str):
    if name in {"PostflopMasks", "PostflopNet"}:
        from pbg.games.poker.model import postflop

        return getattr(postflop, name)
    if name in {"PokerModelConfig", "PokerPublicBatch", "PolicyOutput", "PreflopNet"}:
        from pbg.games.poker.model import preflop

        return getattr(preflop, name)
    raise AttributeError(name)
