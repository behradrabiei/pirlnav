#!/usr/bin/env python3
"""Mean and sample standard deviation for five numbers."""

import statistics

# numbers = [25.17, 24.50, 24.17, 24.50, 22.85]  # Fill with exactly 5 numbers, e.g. [1.0, 2.0, 3.0, 4.0, 5.0]
numbers = [29.19, 22.87, 24.92, 25.63, 22.22]

if len(numbers) != 5:
    raise SystemExit("Set `numbers` to a list of exactly 5 values.")

mean = statistics.mean(numbers)
stdev = statistics.stdev(numbers)  # sample std (N-1 denominator)

print(f"mean = {mean}")
print(f"stdev (sample) = {stdev}")
