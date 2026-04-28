"""PumpFun bonding-curve AMM impact model.

Derives virtual reserves from the observed slot price using the constant-product
invariant K = 30 SOL * 1.073e9 tokens = 3.219e10 (SOL * tokens), then computes
the fill price for a buy / sell of a given size.

Reserves are NOT in the slot_features_60m schema, but the invariant K is
deterministic for every PumpFun token (initial virtual reserves are fixed).
Given current price P:
    v_sol  = sqrt(K * P)              # in SOL
    v_tok  = sqrt(K / P)              # in tokens (decimal-corrected)
A buy of `sol_in` SOL (after the 1% taker fee) moves v_sol up to v_sol+sol_in,
and v_tok = K / new_v_sol. Tokens received = old_v_tok - new_v_tok.

This is the per-trade slip the protocol charges. Layered on top of any
flat slip we already model (priority fee competition, MEV, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass

K_INVARIANT = 30.0 * 1.073e9  # SOL * tokens (initial virtual SOL × initial virtual tokens)
TAKER_FEE = 0.01              # PumpFun protocol fee, 1% per side


@dataclass(frozen=True)
class Fill:
    exec_price: float    # all-in SOL/token paid (or received)
    tokens: float        # tokens received (buy) or sold (sell)
    impact_bps: float    # signed impact in bps relative to spot price


def reserves_from_price(price: float) -> tuple[float, float]:
    """Recover virtual SOL and virtual token reserves from observed spot price.

    price = v_sol / v_tok and v_sol * v_tok = K, so v_sol = sqrt(K*P).
    """
    v_sol = (K_INVARIANT * price) ** 0.5
    v_tok = (K_INVARIANT / price) ** 0.5
    return v_sol, v_tok


def buy_fill(price: float, sol_in: float, fee: float = TAKER_FEE) -> Fill:
    """Simulate a buy of `sol_in` SOL at observed `price`.

    Returns the all-in execution price, tokens received, and slip in bps
    relative to spot. Impact is positive (paid more than spot).
    """
    v_sol, v_tok = reserves_from_price(price)
    sol_to_curve = sol_in * (1.0 - fee)
    new_v_sol = v_sol + sol_to_curve
    new_v_tok = K_INVARIANT / new_v_sol
    tokens_out = v_tok - new_v_tok
    if tokens_out <= 0:
        return Fill(exec_price=float("inf"), tokens=0.0, impact_bps=float("inf"))
    exec_price = sol_in / tokens_out  # full SOL spent (incl fee) per token actually received
    impact_bps = (exec_price / price - 1.0) * 10_000
    return Fill(exec_price=exec_price, tokens=tokens_out, impact_bps=impact_bps)


def sell_fill(price: float, tokens_in: float, fee: float = TAKER_FEE) -> Fill:
    """Simulate a sell of `tokens_in` tokens at observed `price`.

    Returns the all-in execution price (SOL received per token), SOL out,
    and slip in bps relative to spot. Impact is negative (received less than spot).
    """
    v_sol, v_tok = reserves_from_price(price)
    new_v_tok = v_tok + tokens_in
    new_v_sol = K_INVARIANT / new_v_tok
    sol_pre_fee = v_sol - new_v_sol
    sol_out = max(sol_pre_fee * (1.0 - fee), 0.0)
    if tokens_in <= 0:
        return Fill(exec_price=0.0, tokens=0.0, impact_bps=0.0)
    exec_price = sol_out / tokens_in
    impact_bps = (exec_price / price - 1.0) * 10_000  # negative for a sell
    return Fill(exec_price=exec_price, tokens=sol_out, impact_bps=impact_bps)


def round_trip_pnl(entry_price: float, exit_price: float, sol_in: float,
                    fee: float = TAKER_FEE) -> tuple[float, float, float]:
    """Buy `sol_in` SOL at entry_price, sell entire position at exit_price.

    Returns (sol_out, pnl_sol, total_slip_bps). Slip is positive (cost) in bps.
    """
    buy = buy_fill(entry_price, sol_in, fee)
    sell = sell_fill(exit_price, buy.tokens, fee)
    sol_out = sell.tokens  # in sell_fill, tokens field stores SOL out
    pnl = sol_out - sol_in
    no_slip_pnl = sol_in * (exit_price / entry_price - 1)
    slip_bps = (no_slip_pnl - pnl) / max(sol_in, 1e-12) * 10_000
    return sol_out, pnl, slip_bps


if __name__ == "__main__":
    # sanity sample at genesis-like price (slot 0 is post-snipe; pick a few)
    for p in (2.8e-8, 5e-8, 1e-7, 5e-7):
        b = buy_fill(p, 1.0)
        print(f"price={p:.2e}  buy 1 SOL  → exec={b.exec_price:.3e}  "
              f"impact={b.impact_bps:+6.0f} bps  tokens={b.tokens:.2e}")
    print()
    print("Round-trip 1 SOL, 2x price move:")
    for p in (2.8e-8, 1e-7, 5e-7):
        sol_out, pnl, slip = round_trip_pnl(p, 2 * p, 1.0)
        print(f"  entry={p:.2e}  exit={2*p:.2e}  out={sol_out:.4f}  "
              f"pnl={pnl:+.4f}  total_slip={slip:.0f} bps")
