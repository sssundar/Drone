import pstats

# Run python -m cProfile -o profiling_results wiring.py before this
stats = pstats.Stats("profiling_results")
stats.sort_stats("tottime")
stats.print_stats(10)
