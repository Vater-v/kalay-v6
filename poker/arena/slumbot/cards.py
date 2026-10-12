"""Protocol card ids validated and converted at the engine boundary."""

from poker.arena.slumbot.protocol import ProtocolError, our_card
from poker.envs import guards


def engine_cards(hole_ids, board_ids):
    """(hole, board) engine cards of a response's protocol ids, validated on the host (poker.envs.guards.cards): two
    hole cards, a board of 0, 3, 4 or 5 cards, ids in 0..51, no card twice; ProtocolError otherwise (the parser
    checks the strings; this is the engine's own entry, edge_cases C1/C2)."""
    try:
        hole = [our_card(c) for c in guards.cards(hole_ids, "hole card ids", count=2)]
        board = [our_card(c) for c in guards.cards(board_ids, "board card ids")]
        if len(board) not in guards.VISIBLE:
            raise ValueError(f"board = {board}: {len(board)} cards, a street shows {guards.VISIBLE}")
        guards.cards(hole + board, "hole and board cards")
    except ValueError as err:
        raise ProtocolError(f"cards: {err}") from None
    return hole, board
