'''
Directional options strategy backtester (no delta-hedging)
1. Given binary forecast, sell ATM put/call
2. Given continuous forecast, sell OTM put/call w/ delta dependent on forecast magnitude
3. Add OTM protection (put/call spread)

Vol Selling
1. Single-leg strategies: put, call w/ delta hedging (ITM and OTM)
2. Single-leg strategies w/ OTM protection: put spread, call spread
3. Multi-leg strategies: straddle, strangle
4. Multi-leg strategies w/ OTM protection: butterfly, condor
5. Variance swap
'''