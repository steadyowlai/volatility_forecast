Level 1 Features:

features:  
  \- name: spy\_ret\_1d  
    category: returns  
    description: 1-day log return of SPY  
    formula: log(close\_t / close\_t-1)  
    source: curated.market\_daily (SPY)  
    rows: one per trading day

  \- name: spy\_ret\_5d  
    category: returns  
    description: 5-day log return of SPY  
    formula: log(close\_t / close\_t-5)  
    source: curated.market\_daily (SPY)  
    rows: one per trading day (starting day 5\)

  \- name: spy\_ret\_20d  
    category: returns  
    description: 20-day log return of SPY  
    formula: log(close\_t / close\_t-20)  
    source: curated.market\_daily (SPY)  
    rows: one per trading day (starting day 20\)

  \- name: spy\_vol\_5d  
    category: volatility  
    description: 5-day realized volatility of SPY returns  
    formula: sqrt(sum((r\_t)^2 for last 5 days))  
    source: curated.market\_daily (SPY)  
    rows: one per trading day (starting day 5\)

  \- name: spy\_vol\_10d  
    category: volatility  
    description: 10-day realized volatility of SPY returns  
    formula: sqrt(sum((r\_t)^2 for last 10 days))  
    source: curated.market\_daily (SPY)  
    rows: one per trading day (starting day 10\)

  \- name: spy\_vol\_20d  
    category: volatility  
    description: 20-day realized volatility of SPY returns  
    formula: sqrt(sum((r\_t)^2 for last 20 days))  
    source: curated.market\_daily (SPY)  
    rows: one per trading day (starting day 20\)

  \- name: drawdown\_60d  
    category: drawdown  
    description: Peak-to-trough drawdown in last 60 trading days  
    formula: 1 \- (close\_t / max(close\_t-60...t))  
    source: curated.market\_daily (SPY)  
    rows: one per trading day (starting day 60\)

  \- name: vix  
    category: volatility\_index  
    description: CBOE Volatility Index (spot)  
    formula: daily close value  
    source: curated.market\_daily (VIX)  
    rows: one per trading day

  \- name: vix3m  
    category: volatility\_index  
    description: 3-month VIX futures index  
    formula: daily close value  
    source: curated.market\_daily (VIX3M)  
    rows: one per trading day

  \- name: vix\_term  
    category: derived\_ratio  
    description: Term structure ratio \= VIX3M / VIX  
    formula: vix3m / vix  
    source: curated.market\_daily (VIX, VIX3M)  
    rows: one per trading day

  \- name: rsi\_spy\_14  
    category: momentum  
    description: 14-day Relative Strength Index for SPY  
    formula: RSI(14) \= 100 \- 100/(1 \+ RS)  
    source: curated.market\_daily (SPY)  
    rows: one per trading day (starting day 14\)

  \- name: corr\_spy\_tlt\_20d  
    category: correlation  
    description: 20-day rolling correlation between SPY and TLT returns  
    formula: corr(returns\_spy\_20d, returns\_tlt\_20d)  
    source: curated.market\_daily (SPY, TLT)  
    rows: one per trading day (starting day 20\)

  \- name: corr\_spy\_hyg\_20d  
    category: correlation  
    description: 20-day rolling correlation between SPY and HYG returns  
    formula: corr(returns\_spy\_20d, returns\_hyg\_20d)  
    source: curated.market\_daily (SPY, HYG)  
    rows: one per trading day (starting day 20\)

  \- name: hyg\_tlt\_spread  
    category: spread  
    description: Difference in daily return between HYG and TLT  
    formula: ret\_hyg \- ret\_tlt  
    source: curated.market\_daily (HYG, TLT)  
    rows: one per trading day  
