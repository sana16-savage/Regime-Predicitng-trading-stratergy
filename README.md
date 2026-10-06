REGIME PREDICTING TRADING STRATERGY

[us_futures_5_with_stage9.ipynb](https://github.com/user-attachments/files/33113109/us_futures_5_with_stage9.ipynb)

Trading Strategy & Parameters per Regime
The system utilizes a 4-state Hidden Markov Model (HMM) running on a rolling Walk-Forward frame (24×180=4,320 training hours, 24×30=720 test hours) to dynamically classify market regimes using 9 causal features (including log returns, realized volatility, ADX, Hurst exponent, and momentum). Each regime uses a Logistic Regression model conditioned on the underlying dynamics: 
	Mean Reversion Regime: Strategy targets range-bound asset movements where prices bounce within bounds. Features like trailing 100-period Hurst exponent (hurst_100 <0.5) and low ADX are weighted to capture mean-reverting probabilities. 
	Trending Regime: Directional strategy exploiting persistent market moves. Relies heavily on multi-horizon momentum features (momentum_5, momentum_20, momentum_60) and trend strength (adx_14). 
	High Volatility Regime: Designed to manage elevated risk when trailing volatility (log_realized_vol_20) and candle range percentages widen. 
	Event Regime: Designed for extreme market movements, macro announcements, or volume spikes (flagged via volume_zscore_20). 
	Respective Parameters Set:
	HMM Training Window (WF_TRAIN_BARS): 4,320 hourly bars (180 days). 
	HMM Test/Step Window (WF_TEST_BARS / WF_STEP_BARS): 720 hourly bars (30 days). 
	HMM Specifications: 4 final components (HMM_FINAL_N_STATES = 4), diagonal covariance matrix (HMM_COVARIANCE_TYPE = "diag"), max iterations of 800 (HMM_N_ITER = 800), tolerance of 1"e-" 4, and 5 restarts with seed 42. 
2. Why Use Median Win Rate, Median Profit Factor, and Median Max Drawdown?
	Robustness to Outliers: Performance metrics across multiple walk-forward folds (75+ folds) are often heavily skewed by a single hyper-profitable regime or an extreme crash fold.
	Median vs. Mean: Using the median rather than the mean gives a realistic, unskewed baseline of typical fold performance, preventing one or two outlier trades from inflating expected future outcomes.
3. Definitions of Key Thresholds
	Meta Threshold: The master confidence level required across the entire pipeline before executing a trade signal.
	Margin Threshold: The required risk capital or position size cap to avoid over-leveraging during volatile regimes.
	ADX Threshold: The numerical cutoff for the 14-period Average Directional Index (adx_14) to distinguish between a trending market ("ADX">"Threshold" ) and a mean-reverting market ("ADX"<"Threshold" ). 
	Confidence Threshold: The minimum output probability produced by the Logistic Regression classifier (e.g., P("Up" )>0.60) necessary to trigger a buy or sell trade.
4. N Folds with Trade and Passing
	N Folds with Trade: The count of walk-forward test folds out of 75+ where the model generated at least one signal that met all probability/confidence criteria.
	N Folds Passing: The count of folds where the backtested trade sequence met pre-defined minimum performance benchmarks (e.g., positive expected return, acceptable drawdown limits).
5. Strengths & Weaknesses (Pros & Cons) of the Model
	Pros (The Good):
	Leak-Free Pipeline: Features are engineered strictly without lookahead bias using rolling windows. 
	Adaptability: Walk-forward re-training prevents alpha decay by adapting to changing market dynamics every 30 days. 
	Context-Aware: HMM regime filtering prevents momentum models from trading in range-bound markets and vice-versa. 

1. Data Ingestion parameter from Binance Api key
<img width="708" height="379" alt="image" src="https://github.com/user-attachments/assets/3b70bebe-9ca7-4e03-80d9-7abdfb53b5b3" />
<img width="311" height="379" alt="image" src="https://github.com/user-attachments/assets/1a8afdf3-659a-4595-8a1b-49c1030a9718" />
<img width="671" height="276" alt="image" src="https://github.com/user-attachments/assets/cc95b9dd-f6d1-4d56-ba80-1e2370941ed0" />

2. HMM Pattern regonition stage
Walk-forward HMM
The HMM isn't trained once on the entire dataset.
The notebook uses:
Training = 180 days
Testing  = 30 days
Step     = 30 days
for the Stage 5 regime detection process. 
Conceptually:
Train 180 days → Test 30 days
       ↓
Train 180 days → Test 30 days
             ↓
Train 180 days → Test 30 days
                       ...
The important point is:
the future test period isn't used to fit the model.
That is what makes it walk-forward/out-of-sample leaving no overfit to model.
The model works fully on real life data giving good predictions as no future value is leaked
while training the model as we shift the next 4 bars are moved back for model to guess it and the check its predictions with historical data 


<img width="436" height="338" alt="image" src="https://github.com/user-attachments/assets/0288c5b1-cff5-48eb-b3d1-446b92d1665e" />
<img width="309" height="387" alt="image" src="https://github.com/user-attachments/assets/b10e26b7-787f-4e40-b354-fbcba7d30d5a" />


Future-regime target
This stage changes the question.
Stage 5 tells us:
"What regime are we in now?"
Stage 6 creates:
"What regime will we be in in the future?"
Your target uses:
t+1
t+2
t+3
t+4
future regime states.
For example:
Current time t
       │
       ▼
Current regime = Trending

Future:
t+1 = Trending
t+2 = Trending
t+3 = Trending
t+4 = Trending

Target = Trending
The Stage 6 output contains:
future_state_t_plus_1
future_state_t_plus_2
future_state_t_plus_3
future_state_t_plus_4
future_regime
future_regime_label
For BTC, Stage 6 produced 53,973 usable rows after removing observations without the complete future target. 
________________________________________
Stage 7 — LightGBM future-regime classifier
Now the machine-learning model is trained.
The question becomes:
"Given everything I know at time t, what regime will the market be in the future?"
The model predicts four classes:
Trending
Mean Reversion
High Volatility
Event
The model produces probabilities such as:
Trending           0.72
Mean Reversion     0.12
High Volatility    0.10
Event              0.06
Then:
predicted regime = Trending
prediction confidence = 0.72
The notebook defines prediction confidence from the maximum class probability. 
So 0.72 does NOT mean a 72% probability of a LONG trade.
It means approximately:
"The classifier assigns 72% probability to the predicted regime."
This distinction is very important.
Stage 7 uses LightGBM with walk-forward training rather than training once on the entire history. 
The reported OOS results include, for example:
BTC accuracy ≈ 87.34%
ETH accuracy ≈ 88.50%
from the notebook's Stage 7 output. 
________________________________________

Stage 8 — actual trading strategy
This is where the system finally converts:
predicted regime
       +
market features
       +
confidence
       +
meta-model
into:
LONG
SHORT
NO TRADE
There are two layers.
3. LOGISTIC REGRESSION WITH ITERATION OF TOOLS to find the edge on with bitcoin we can get using this model with most appropriate parameter

a.META-LABELING
   The rule-based regime signal (Trending momentum/ADX, Mean-Reversion
   EMA-fade) is kept as the "primary model" that proposes WHEN and WHICH
   DIRECTION to trade. A secondary binary classifier (meta-model) is trained
   to predict whether that specific proposed trade will actually win, using
   only information available at signal time (regime probabilities, ADX,
   Hurst, multi-scale momentum agreement, volatility z-score, distance from
   EMA in ATR units). Only trades the meta-model is confident about are
   taken. This is the direct lever for lifting the realized win rate above
   52%: it trades less often, but only on the setups the data says look
   like winners.

b. MULTI-FOLD WALK-FORWARD (not one split)
   The data is split into N rolling folds (train -> embargo -> test). The
   meta-model is refit on each fold's training window only, and every
   config is scored on ALL test folds. A config only "passes" if it clears
   the win-rate bar on a majority of folds, not just one.

c. CONFLUENCE FILTERS PER REGIME
   Trending trades require multi-timeframe momentum agreement (5h/20h/60h
   same sign) AND Hurst > 0.5 (the market is actually persistent). Mean
   reversion trades require Hurst < 0.5 (actually mean-reverting) AND
   volume not exploding (an extension driven by a volume surge is a
   breakout, not something to fade).

d. HONEST OUTPUT
   For each symbol, if no config clears win_rate >= 0.52 on a majority of
   folds with profit_factor > 1, the script says so explicitly instead of
   reporting the least-bad option as if it were a strategy. Backtested
   "best of a grid" numbers that fail out-of-sample validation are not an
   edge, they're a Sharpe-ratio-shape.
   
<img width="778" height="357" alt="image" src="https://github.com/user-attachments/assets/02f1d6e0-3e49-4fab-9caa-a3722a5d4f66" />
<img width="803" height="373" alt="image" src="https://github.com/user-attachments/assets/8616101f-f2bd-402d-aa3e-353d0c811a89" />
<img width="737" height="149" alt="image" src="https://github.com/user-attachments/assets/f99313bc-9b5e-4c64-b8fd-6fe5372d5b10" />
<img width="788" height="89" alt="image" src="https://github.com/user-attachments/assets/45e542f2-2b58-4d02-acc3-21f1ef081c1b" />

WE SUCCESSFULLY FOUND EDGE IN ETHIRIUM.
I'll soon post the walk forward and Live trade results. stay tuned


