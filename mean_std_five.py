#!/usr/bin/env python3
"""Mean and sample standard deviation for five numbers."""

import statistics

# numbers = [25.17, 24.50, 24.17, 24.50, 22.85]  # Fill with exactly 5 numbers, e.g. [1.0, 2.0, 3.0, 4.0, 5.0]
success_rates = [43.40, 37.74, 43.40, 39.62, 35.85]
spls = [17.01, 13.59, 16.12, 16.06, 15.31]


if len(success_rates) != 5 or len(spls) != 5:
    raise SystemExit("Set `numbers` or `spls` to a list of exactly 5 values.")

mean_success_rate = statistics.mean(success_rates)
stdev_success_rate = statistics.stdev(success_rates)  # sample std (N-1 denominator)

mean_spl = statistics.mean(spls)
stdev_spl = statistics.stdev(spls)  # sample std (N-1 denominator)

print(f"mean success rate = {mean_success_rate}")
print(f"stdev success rate = {stdev_success_rate}")
print(f"mean SPL = {mean_spl}")
print(f"stdev SPL = {stdev_spl}")
