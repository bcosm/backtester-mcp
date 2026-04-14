"""One-shot benchmark for README. Prints Markdown table of real timings on this machine."""
import time
import numpy as np
from backtester_mcp import backtest

def make_inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    prices = np.cumprod(1 + rng.normal(0, 0.01, n))
    # alternating-enough signals so trades actually fire
    fast = np.convolve(prices, np.ones(10) / 10, mode="full")[:n]
    slow = np.convolve(prices, np.ones(50) / 50, mode="full")[:n]
    signals = np.where(fast > slow, 1.0, -1.0)
    signals[:50] = 0.0
    return prices, signals

def bench(n, repeats=5, warmup=True):
    prices, signals = make_inputs(n)
    if warmup:
        _ = backtest(prices, signals)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = backtest(prices, signals)
        times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times), max(times)

def main():
    sizes = [1_000, 10_000, 100_000, 500_000]

    # cold run (first call includes JIT)
    n = 10_000
    prices, signals = make_inputs(n)
    t0 = time.perf_counter()
    _ = backtest(prices, signals)
    cold = time.perf_counter() - t0
    print(f"COLD run (10k bars, first call with JIT): {cold*1000:.1f} ms")

    print("\n| Bars | Time (min of 5) |")
    print("|---|---|")
    for s in sizes:
        mn, avg, mx = bench(s, repeats=5)
        if mn < 0.001:
            t_str = f"~{mn*1_000_000:.0f} us"
        elif mn < 1.0:
            t_str = f"~{mn*1000:.1f} ms"
        else:
            t_str = f"~{mn:.2f} s"
        print(f"| {s:,} | {t_str} |  (min {mn*1000:.3f} ms, avg {avg*1000:.3f} ms, max {mx*1000:.3f} ms)")

if __name__ == "__main__":
    main()
