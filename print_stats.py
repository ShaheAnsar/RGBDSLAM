import sys
import pstats
from pstats import SortKey


s = pstats.Stats("stats")
if len(sys.argv) == 1:
    s.sort_stats(SortKey.CUMULATIVE).print_stats(20)
else:
    s.sort_stats(SortKey.CUMULATIVE).print_stats(int(sys.argv[1]))
