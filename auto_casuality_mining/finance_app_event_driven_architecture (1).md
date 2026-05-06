# Event-Driven Finance App Architecture, Signal Catalog, and Roadmap

## Executive Summary

The product should be treated as a real-time article decisioning system whose job is to identify market-relevant moments, personalize them against each user’s portfolio and behavioral context, and select the article angle most likely to produce a click and a subsequent trade. The core system loop is: ingest event, enrich it with user and portfolio context, infer which user clusters are receptive, generate article candidates, rank and schedule them, measure outcomes, and feed results back into the causal graph inference layer.

The central technical bet is not simply “better content.” It is learning which signals, article frames, timing windows, and user states create incremental trade intent for each cluster. CTR is the near-term signal, but the target should be a composite action objective that captures click, article depth, watchlist action, order ticket open, and executed trade within a defined attribution window.

The signal roadmap should start with public, low-friction data such as price movement, earnings, news, corporate actions, analyst notes, and user holdings. It should then move into higher-value proprietary signals such as in-app behavior, portfolio similarity cohorts, social/network momentum, friend-circle themes, and internal house views. The hardest but most differentiated features require consented brokerage data, social graph data, private behavioral data, and causal inference infrastructure that can separate correlation from incremental influence.

## 1. Solution Architecture and Building Blocks

### 1.1 Product Objective

The system optimizes for timely personalized content that increases article CTR and downstream trading activity. The base metric should not be raw trade count alone, because that can reward low-quality triggering and churn-risk behavior. A stronger optimization target is:

\[
OpportunityScore = w_1 CTR + w_2 ReadDepth + w_3 OrderIntent + w_4 ExecutedTrade + w_5 RetentionAdjustedValue
\]

For growth experimentation, the product can start with CTR and presence of trade, but the data model should capture intermediate intent events from day one. A good event hierarchy is impression, notification open, article click, article read depth, CTA click, asset page view, watchlist add, order ticket open, order submitted, order executed, and post-trade retention behavior.

### 1.2 High-Level Architecture

The architecture should be event-driven with the following layers:

1. **Signal ingestion layer** receives public and private events: price moves, news, disclosures, analyst changes, earnings, macro events, social trends, portfolio changes, user behavior, and internal house views.
2. **Event normalization and entity resolution layer** maps every event to canonical entities such as ticker, issuer, sector, geography, theme, factor, peer group, portfolio exposure, and event type.
3. **Real-time feature store** maintains user, asset, portfolio, cluster, and market-state features with timestamped values for offline training and online serving.
4. **User clustering layer** groups users by portfolio composition, trading style, sensitivity to event types, risk appetite proxies, content preferences, recency of activity, and social/network context.
5. **Causal graph inference layer** estimates which signals and article treatments are likely to cause incremental clicks and trades for each user cluster.
6. **Article opportunity generator** converts eligible events into article candidates, each with a hypothesis, target cluster, asset universe, article frame, urgency level, and CTA.
7. **Content assembly layer** creates personalized articles from templates, retrieval, generated narrative, charts, portfolio context, and house view.
8. **Ranking, timing, and pacing layer** chooses which article to publish, through which channel, at what time, and with which headline or notification copy.
9. **Experimentation and attribution layer** tracks treatment exposure, holdouts, counterfactual estimates, and downstream actions.
10. **Feedback loop** updates causal graph parameters, cluster definitions, article policies, and exploration budgets.

### 1.3 Event Flow

The core event flow should look like this:

```
External/Internal Signal
    -> Event Bus
    -> Entity Resolution
    -> Signal Scoring
    -> User/Portfolio Matching
    -> Cluster-Level Causal Inference
    -> Article Candidate Generation
    -> Article Ranking and Timing
    -> Content Generation
    -> Delivery
    -> Outcome Tracking
    -> Attribution and Model Update
```

The most important design decision is that the system should not generate one article per event. It should generate many article candidates per event, each representing a different causal hypothesis. For example, a semiconductor earnings surprise could become a “your holding is exposed,” “peer trade opportunity,” “sector rotation,” “AI infrastructure momentum,” “risk alert,” or “friends/cohort attention” article. The model should learn which framing works for which cluster.

### 1.4 Core Data Objects

The system should standardize the following objects.

**SignalEvent** represents something that happened. Fields should include event ID, event type, source, timestamp, latency, affected tickers, affected sectors, themes, sentiment, magnitude, novelty, urgency, confidence, and source reliability.

**UserState** represents the current user context. Fields should include holdings, watchlist, cash balance band, recent trades, article engagement history, preferred assets, preferred topics, typical trading time, sensitivity to gains/losses, recent inactivity, and cluster membership.

**PortfolioExposure** represents how an event maps to a user. Fields should include direct holding exposure, indirect sector exposure, factor exposure, peer exposure, currency exposure, country exposure, and unrealized gain/loss context.

**ArticleCandidate** represents a possible article. Fields should include trigger event, target user cluster, article frame, headline family, body template, personalized inserts, CTA, urgency level, expected CTR, expected trade probability, expected incremental lift, and suppression reason if not delivered.

**TreatmentExposure** represents what the user was shown. Fields should include article ID, headline variant, channel, send time, rank position, notification copy, CTA, user cluster at exposure time, and competing articles suppressed.

**OutcomeEvent** captures feedback. Fields should include impression, click, scroll depth, dwell time, CTA click, asset page view, watchlist action, order ticket open, trade submitted, trade executed, trade size band, trade side, and attribution window.

### 1.5 Causal Graph Inference Layer

The causal graph should model the relationship between user state, signal type, article treatment, timing, and outcomes. Its job is to decide which signal should be used to create an article, not merely which observed correlations are strongest.

At minimum, the graph should include these node families:

- **User cluster nodes**: portfolio archetype, activity frequency, responsiveness, trading sophistication proxy, preferred content format, prior engagement, recent loss/gain state.
- **Portfolio context nodes**: direct holding, watchlist exposure, sector concentration, recent P&L, unrealized gain/loss, cash availability band, position size band.
- **Market context nodes**: volatility regime, market trend, sector momentum, macro calendar, liquidity, market open/close proximity.
- **Signal nodes**: news type, price move, earnings, analyst rating, corporate filing, social trend, peer/network activity, house view update.
- **Treatment nodes**: article frame, headline, CTA, personalization depth, notification copy, channel, time delay after signal, frequency cap status.
- **Outcome nodes**: click, read depth, CTA click, order ticket open, trade execution, repeat engagement, churn proxy, mute/unsubscribe.
- **Confounder nodes**: user activeness, market-wide trend, asset popularity, day/time, prior intent, portfolio size band, prior watchlist interest.

The causal layer should distinguish three concepts:

1. **Prediction**: “Who is likely to click or trade?”
2. **Uplift**: “Who is more likely to click or trade because we showed this article?”
3. **Policy selection**: “Which article, timing, and channel should we choose given limited attention budget?”

The product should optimize on uplift, not raw propensity. If a user would have traded anyway because they were already viewing the asset page, publishing an article may show high conversion but low incrementality. The model should reserve article inventory for moments where content changes behavior.

### 1.6 Learning Approach

The system should use causal graph inference from day one. The graph does not need to be “complete” on day one; it should start as a sparse causal structure with explicit priors, rules, and uncertainty. As more data becomes available, the same graph becomes more granular, more personalized, and more confident.

The important principle is that the architecture stays broadly the same across the roadmap. What changes is the amount of observable context available to the graph. With only public market data, the graph can infer coarse relationships between market events, article frames, CTR, and trade outcomes. With portfolio data, it can infer whether the event matters to the user. With behavioral data, it can infer readiness and intent. With social, network, and house-view data, it can infer what angle, CTA, and timing are most likely to create action.

The causal graph should therefore operate in three modes at all times:

1. **Structural prior mode**: When data is sparse, the model uses domain priors and deterministic relationships. For example, a direct holding should causally increase relevance, a fresh earnings surprise should increase urgency, and a recent asset page view should increase trade readiness.
2. **Observed evidence mode**: As click, read, order, and trade data accumulates, the model updates edge weights between signals, treatments, user states, and outcomes. For example, it may learn that analyst upgrades drive CTR for one cluster but only price-move alerts drive trade activity for another.
3. **Intervention learning mode**: Through holdouts, delayed sends, headline variants, frame tests, and channel tests, the model estimates incremental lift rather than raw correlation. This lets it learn whether the article actually changed behavior or merely appeared before a trade the user was already going to make.

The model’s output should always include both a decision and a confidence level. Early on, confidence may be low and the system should rely more heavily on conservative relevance and urgency rules. Later, as the graph sees more treatment variation and outcome data, it can make finer decisions about signal selection, article framing, CTA, send time, and sequencing.

#### Causal Graph Discovery and Refutation Pipeline

The causal graph should be continuously proposed, challenged, and refreshed through an automated discovery pipeline. This is the mechanism that turns raw time-series data into candidate causal relationships.

First, all data should be normalized into time series at a consistent decisioning grain. Depending on the signal, the grain may be one minute, five minutes, hourly, daily, or event-relative time. Every market signal, user behavior signal, article exposure, CTA action, order event, trade event, portfolio state, social signal, and house-view update should be represented as a timestamped feature. The feature store should preserve both raw event time and model-ready aligned time buckets.

Second, for each target variable, the system should train a range of predictive models using other time-series variables with small lag shifts. For example, the system can ask whether a price move, news event, article frame, watchlist action, social attention spike, or prior asset page view at time \(t - \Delta\) helps predict CTR, order ticket opens, or trade activity at time \(t\). The lag shift is important because it forces the model to learn temporally plausible relationships rather than same-timestamp leakage.

Third, the system should use feature-importance methods such as SHAP values to identify which lagged features are consistently important for predicting each target variable. These important features should not be treated as causal proof. They should be treated as candidate edges in the proposed causal graph. For example, if “volume spike at \(t - 15m\), article exposure at \(t - 10m\), and recent asset page view at \(t - 1h\)” consistently predict “order ticket open at \(t\),” the pipeline can propose edges from those variables into the order-intent node.

Fourth, the proposed graph should go through an automated causal refutation pipeline before production use. Candidate edges should be challenged with temporal holdouts, placebo tests, delayed-treatment tests, negative controls, sensitivity checks for confounders, stability tests across user clusters, and out-of-sample validation. Edges that are predictive but unstable, directionally implausible, or explainable by confounders should be rejected or kept as low-confidence features rather than promoted into the production causal graph.

Fifth, only graph relationships that survive refutation should be promoted into production decisioning. Promotion should include metadata: edge strength, confidence interval, supported clusters, supported lag window, last validation date, known confounders, and applicable signal types. This lets the article policy service know not just that a relationship exists, but where and when it is safe to use.

This process should repeat weekly. Every week, the system should rebuild candidate time-series features, retrain the predictive model set, recompute SHAP-based candidate edges, rerun causal refutation, compare the new graph with the current production graph, and promote only relationships that improve prediction and survive refutation. The weekly cadence gives the model enough data to detect behavior changes while avoiding overreaction to daily noise.

The learning goal also changes by data availability. At low data maturity, the goal is to select timely articles that are plausibly relevant. At medium data maturity, the goal is to personalize article quality: better content, better explanation, better CTA, better timing, and better matching of article frame to user cluster. At high data maturity, the goal is to create new user behavior by surfacing better opportunities at better moments. The system may stay technically similar, but user behavior can change because the content becomes more useful, more specific, and more action-oriented.

Contextual bandits and reinforcement learning should be viewed as decision policies layered on top of the causal graph, not as replacements for it. The graph defines what is believed to cause CTR and trade activity; bandits decide how much to explore competing article variants; sequence policies decide how to guide a user across multiple articles, asset views, watchlist actions, and order moments.

### 1.7 Article Decisioning Policy

Each candidate article should be scored using:

\[
ArticleScore = ExpectedIncrementalCTR \times ExpectedTradeLift \times Urgency \times Relevance \times Freshness \times Confidence \times InventoryFit
\]

Where:

- **ExpectedIncrementalCTR** estimates causal lift in click probability.
- **ExpectedTradeLift** estimates incremental probability of trade within the attribution window.
- **Urgency** captures time sensitivity of the market event.
- **Relevance** captures portfolio, watchlist, and interest match.
- **Freshness** penalizes stale events.
- **Confidence** reflects source reliability and model confidence.
- **InventoryFit** accounts for notification fatigue, frequency caps, and competing opportunities.

The ranking layer should produce not only a top article, but also a reason code. Example reason codes: “direct holding exposed to earnings surprise,” “watchlist asset crossing momentum threshold,” “sector peer moving after news,” “cluster responds to analyst upgrade frames,” or “network cohort attention spike.”

### 1.8 Personalization Strategy

Personalization should happen at four levels:

1. **Selection personalization**: decide whether the event is relevant to the user.
2. **Framing personalization**: decide whether to present the article as opportunity, risk, comparison, earnings, momentum, valuation, theme, or peer activity.
3. **Content personalization**: insert relevant holdings, watchlist names, sector exposure, similar assets, and recent price context.
4. **CTA personalization**: choose between read more, view asset, compare peers, add to watchlist, review exposure, open order ticket, or follow theme.

The system should not always push directly to trade. Sometimes the best conversion path is “article -> asset page -> watchlist -> price alert -> later trade.” Capturing these intermediate states improves model learning and prevents overfitting to immediate trades only.

### 1.9 User Clustering

User clusters should be refreshed periodically and also updated in near real time for major behavior changes. Useful cluster families include:

- **Portfolio archetype**: concentrated tech, dividend income, ETF-heavy, meme-stock active, crypto-adjacent, China/HK exposure, US mega-cap, sector specialist.
- **Trading style**: frequent trader, event trader, dip buyer, momentum chaser, long-term holder, earnings trader, options user if applicable.
- **Content response style**: clicks on headlines, reads long-form, reacts to charts, prefers bullet summaries, responds to urgency, responds to peer comparisons.
- **Risk state**: sitting on large gains, sitting on losses, cash-rich, fully invested, recent drawdown, recent profit.
- **Lifecycle stage**: newly onboarded, connected portfolio but inactive, first-trade candidate, retained active trader, lapsed user.
- **Network context**: has active friends, cohort with high recent trading activity, belongs to a theme-following group, influenced by leaderboard or community momentum.

Clusters should be designed for decisioning, not just analytics. A good cluster is one where article policy differs materially.

### 1.10 Feedback Signals and Attribution

The feedback loop should track at least three attribution windows:

- **Immediate window**: 0-2 hours after exposure. Useful for urgent price/news triggers.
- **Short window**: same day. Useful for earnings, analyst changes, and market moves.
- **Medium window**: 1-7 days. Useful for thematic articles and research-led ideas.

Outcomes should be attributed using both direct and incremental methods. Direct attribution asks whether the user clicked and traded after exposure. Incremental attribution asks whether exposed users traded more than similar unexposed or delayed-exposure users.

The minimum viable experimentation setup should include:

- Randomized holdout by cluster and signal type.
- Delayed-send controls for time-sensitive events.
- Headline and frame experiments.
- Channel experiments across push, email, in-app feed, and asset page modules.
- Frequency-cap experiments to find the point where incremental trades are offset by fatigue.

### 1.11 Metrics

The primary growth metrics should be:

- Article CTR by signal type, cluster, channel, and timing window.
- Click-to-trade conversion within defined windows.
- Incremental trade lift versus holdout.
- Trade value or revenue proxy per article impression.
- Order ticket open rate.
- Asset page view rate after article click.
- Watchlist add rate.
- Repeat article engagement within 7 and 30 days.

Diagnostic metrics should include:

- Signal-to-article latency.
- Article generation success rate.
- Ranking confidence distribution.
- Suppression rate by reason.
- Notification fatigue rate.
- Cluster drift.
- Model calibration by cluster.
- False-positive trigger rate, meaning articles with high predicted score but low realized engagement.

### 1.12 Operational Building Blocks

The minimum engineering stack should include:

- Event bus for market, content, user, and portfolio events.
- Canonical security master and entity resolution service.
- Real-time and offline feature stores.
- Portfolio exposure engine.
- User clustering service.
- Signal scoring service.
- Article candidate generator.
- Content generation and template engine.
- Retrieval layer for market/news/house-view context.
- Ranking and policy service.
- Delivery orchestration service.
- Experimentation platform.
- Outcome attribution pipeline.
- Time-series normalization pipeline.
- Candidate causal edge discovery pipeline.
- SHAP-based feature importance pipeline.
- Automated causal refutation pipeline.
- Weekly graph promotion and rollback workflow.
- Model monitoring and retraining pipeline.
- Analyst/operator console for inspecting triggers, suppressions, and article performance.

## 2. Comprehensive Signal Catalog

This catalog is organized by signal family. Each signal can be used as a trigger, ranking feature, personalization feature, or causal graph node.

### 2.1 Market Price and Liquidity Signals

These are usually the easiest external signals to acquire and among the strongest for timely triggers.

| Signal | Use Case | Personalization Angle | Acquisition Difficulty |
|---|---|---|---|
| Intraday price move | Trigger urgency articles | “Your holding is moving” or “watchlist breakout” | Low |
| Gap up/down at open | Morning action trigger | Direct holding or sector peer move | Low |
| 52-week high/low | Momentum or reversal frame | “Your stock is approaching a key level” | Low |
| Moving average crossover | Technical momentum article | Momentum-sensitive clusters | Low |
| Relative strength vs index | Outperformance/underperformance | Compare user holding to benchmark | Low |
| Sector-relative move | Sector rotation trigger | Portfolio sector concentration | Low |
| Volume spike | Attention/liquidity event | “Unusual activity around this name” | Low |
| Volatility spike | Risk or opportunity article | Risk-sensitive and options-active clusters | Low |
| Bid-ask spread widening | Liquidity/risk alert | Thinly traded holdings | Medium |
| Options volume spike | Speculation/attention signal | Active traders, options-interested users | Medium |
| Put/call ratio | Sentiment or hedging narrative | Contrarian or event-driven clusters | Medium |
| Short interest change | Squeeze or bearish pressure story | Meme/short-squeeze clusters | Medium |
| Borrow rate change | Crowded short signal | Advanced trader clusters | Medium |
| ETF flow into sector/theme | Theme momentum article | Users exposed to sector/theme | Medium |
| Futures move pre-market | Macro or opening urgency | Index/ETF holders | Low-Medium |
| FX move | Currency exposure trigger | Users holding ADRs, foreign stocks, HK/US cross-border names | Low |
| Crypto price move | Risk-on/risk-off or crypto-linked equity trigger | Crypto-adjacent users | Low |

### 2.2 Corporate Announcement Signals

These are high-conversion because they are directly tied to security-specific action.

| Signal | Use Case | Personalization Angle | Acquisition Difficulty |
|---|---|---|---|
| Earnings release | Immediate event article | Direct holder/watchlist exposure | Low-Medium |
| Earnings surprise | Beat/miss article | “What changed vs expectations” | Medium |
| Revenue guidance | Forward-looking catalyst | Growth-stock clusters | Medium |
| Margin guidance | Quality/profitability frame | Fundamental investors | Medium |
| Buyback announcement | Capital return article | Dividend/value clusters | Low-Medium |
| Dividend declaration/change | Income article | Income portfolio cluster | Low |
| Stock split | Accessibility/momentum article | Retail-heavy clusters | Low |
| M&A announcement | Event-driven trade trigger | Holder, peer, sector exposure | Low-Medium |
| Spin-off/restructuring | Special situation article | Advanced users | Medium |
| Management change | Narrative reset article | Fundamental/news-sensitive clusters | Low-Medium |
| Product launch | Growth/theme article | Thematic users | Low |
| Regulatory approval/rejection | Binary catalyst | Healthcare/biotech holders | Medium |
| Lawsuit/regulatory action | Risk article | Direct and peer exposure | Medium |
| Insider buying/selling | Confidence or concern frame | Fundamental and sentiment clusters | Medium |
| SEC/HKEX/filing changes | Disclosure trigger | Direct holding, large-cap holders | Medium |
| Share placement/dilution | Downside/risk article | Small-cap holders | Medium |
| Debt issuance/rating action | Balance sheet risk frame | Credit-sensitive users | Medium |

### 2.3 News and Media Signals

News signals are plentiful but noisy. They need novelty, source reliability, and portfolio relevance scoring.

| Signal | Use Case | Personalization Angle | Acquisition Difficulty |
|---|---|---|---|
| Breaking news | Timely trigger | Direct holding/watchlist relevance | Low-Medium |
| News sentiment shift | Ranking feature | Positive/negative surprise frame | Medium |
| Coverage volume spike | Attention trigger | “Why everyone is watching this stock” | Medium |
| Reputable source publication | Trust-enhancing article | Source-aware users | Low-Medium |
| Local language news | Regional relevance | HK/China/US market segmentation | Medium |
| CEO interview | Narrative catalyst | Founder/management-sensitive clusters | Medium |
| Product review/news | Consumer demand proxy | Consumer stock holders | Medium |
| Supply chain news | Indirect exposure article | Supplier/customer mapping | High |
| Geopolitical headline | Macro risk trigger | Country/sector exposure | Medium |
| Regulatory rumor/news | Risk/opportunity trigger | Sector-specific clusters | Medium |
| Competitor news | Peer comparison | “What this means for your holding” | Medium |

### 2.4 Analyst, Research, and Institutional Signals

These work well for credibility and “permission to act” framing.

| Signal | Use Case | Personalization Angle | Acquisition Difficulty |
|---|---|---|---|
| Analyst upgrade/downgrade | Directional trigger | Holder/watchlist action | Medium |
| Price target change | Upside/downside article | Compare target to current price | Medium |
| Consensus estimate revision | Forward-looking article | Earnings-sensitive users | Medium |
| Initiation of coverage | Discovery article | Watchlist/theme followers | Medium |
| Institutional ownership change | Smart-money frame | Fundamental clusters | Medium-High |
| Fund 13F/HK disclosure change | Follow-the-money content | Users interested in famous investors | Medium-High |
| Short seller report | Risk/controversy article | Holders and high-vol traders | Medium |
| Index inclusion/removal estimate | Flow catalyst | Index/ETF-aware clusters | Medium |
| Credit rating change | Balance sheet article | Value/dividend clusters | Medium |

### 2.5 Macro and Cross-Asset Signals

These are most useful when mapped to portfolio exposures.

| Signal | Use Case | Personalization Angle | Acquisition Difficulty |
|---|---|---|---|
| CPI/inflation release | Macro-to-portfolio article | Rate-sensitive holdings | Low |
| Fed/central bank decision | Market regime trigger | Growth vs value exposure | Low |
| Yield curve move | Banking, real estate, growth impact | Sector exposure | Low |
| Commodity price move | Energy/materials trigger | Sector and supplier exposure | Low |
| Oil price move | Energy, airlines, consumer impact | Direct and indirect holdings | Low |
| Gold move | Risk-off narrative | Defensive investor clusters | Low |
| USD/CNY/HKD movement | FX exposure | ADRs, exporters, HK users | Low |
| Unemployment/jobs data | Macro sentiment | Index/ETF holders | Low |
| PMI/industrial data | Cyclical exposure | Industrial/materials portfolios | Low |
| Country policy announcement | Regional market catalyst | China/HK/US exposure | Medium |

### 2.6 Social, Viral, and Attention Signals

These are powerful for CTR because they create urgency and curiosity. They need careful normalization because raw virality can be noisy.

| Signal | Use Case | Personalization Angle | Acquisition Difficulty |
|---|---|---|---|
| X/Twitter mention spike | Viral attention article | Meme/active trader clusters | Medium |
| Reddit mention spike | Retail momentum trigger | Meme-stock clusters | Medium |
| YouTube/creator coverage | Narrative diffusion | Users who follow creator-like content | Medium |
| TikTok/short-video trend | Retail attention trigger | Younger/social clusters | Medium |
| Search trend spike | Curiosity and discovery | Broad user interest | Medium |
| Finance forum discussion spike | Early retail attention | Active traders | Medium |
| Influencer bullish/bearish post | Commentary article | Users who react to social proof | Medium |
| Trending keyword/theme | Theme article | Thematic portfolio match | Medium |
| Comment sentiment shift | Momentum/controversy | Socially influenced clusters | Medium-High |
| Meme propagation velocity | FOMO-style article | High-volatility clusters | High |

### 2.7 Portfolio-Derived Signals

These are high-value because they personalize public events into “this matters to you.”

| Signal | Use Case | Personalization Angle | Acquisition Difficulty |
|---|---|---|---|
| Direct holding exposure | Core personalization | “This affects a stock you own” | Medium |
| Watchlist exposure | Conversion trigger | “A stock you follow is moving” | Medium |
| Sector concentration | Risk/opportunity article | “Your portfolio is exposed to this sector” | Medium |
| Unrealized gain | Profit-taking or momentum frame | “You are sitting on gains” | Medium |
| Unrealized loss | Recovery/risk frame | “What today’s news means after recent drawdown” | Medium |
| Position size band | Prioritize relevance | Bigger positions get higher urgency | Medium |
| Cash availability band | Trade readiness feature | Higher likelihood of action | Medium |
| Recent buy | Reinforcement/update article | “New development since you bought” | Medium |
| Recent sell | Re-entry or validation article | “What happened after you exited” | Medium |
| Diversification gap | Recommendation frame | “You lack exposure to this theme” | Medium |
| Concentration risk | Risk article | Overweight names/sectors | Medium |
| Correlation between holdings | Portfolio construction article | Similar holdings affected by same event | High |
| Tax lot/profit context | Realized action angle | Advanced personalization | High |

### 2.8 In-App Behavioral Signals

These are among the highest-return proprietary signals because they identify intent before trade.

| Signal | Use Case | Personalization Angle | Acquisition Difficulty |
|---|---|---|---|
| Article impressions | Fatigue and ranking | Avoid repeat overexposure | Low |
| Article clicks | Interest model | Topic and frame preference | Low |
| Dwell time/read depth | Content quality feedback | Long-form vs short-form preference | Low |
| Asset page views | Pre-trade intent | Follow-up article trigger | Low |
| Search queries | Explicit intent | Article generation on searched tickers/themes | Low-Medium |
| Watchlist add/remove | Intent and cooling-off signal | Follow-up triggers | Low |
| Alert creation | Intent strength | Timed article around alert thresholds | Low |
| Order ticket open | High-intent signal | Retarget with article or reminder | Medium |
| Order abandonment | Objection signal | Article addressing catalyst or risk | Medium |
| Portfolio review frequency | Engagement state | Higher content tolerance | Low |
| Notification opens/mutes | Channel optimization | Timing/frequency policy | Low |
| Preferred reading time | Send-time optimization | Personalized delivery | Low |
| Scroll stopping point | Article format learning | Summaries vs detail | Medium |
| CTA click type | Funnel preference | Compare, chart, order, watchlist | Low |

### 2.9 Brokerage and Trading Signals

These require account connection but directly support conversion modeling.

| Signal | Use Case | Personalization Angle | Acquisition Difficulty |
|---|---|---|---|
| Holdings | Base personalization | Direct exposure | Medium |
| Trades | Outcome label and behavior profile | Trading style clustering | Medium |
| Trade frequency | Frequency cap and ranking | Active vs passive policy | Medium |
| Trade side | Buy/sell propensity | Frame selection | Medium |
| Average trade size band | Value prediction | Revenue-weighted ranking | Medium |
| Asset class usage | Product personalization | Stocks, ETFs, options, crypto | Medium |
| Account value band | Segmentation | High-value user prioritization | Medium |
| Cash balance band | Trade readiness | Actionable opportunity scoring | Medium |
| Margin usage | Risk/trading sophistication | Advanced trader cluster | High |
| Options approval/use | Options-specific triggers | Volatility/event articles | High |
| Transfer/deposit event | Trade readiness | “Cash deployed?” articles | Medium-High |
| Dividend received | Reinvestment trigger | Income clusters | Medium |

### 2.10 Network, Friend, and Cohort Signals

These are potentially powerful because they introduce social proof and conversation utility. The product should avoid exposing private friend-level trading details in article text. The useful version is to convert individual activity into consented, aggregated, or cohort-level momentum features.

| Signal | Use Case | Personalization Angle | Acquisition Difficulty |
|---|---|---|---|
| Friends are discussing a ticker | Social relevance trigger | “A name gaining attention in your circle” | High |
| Friends added to watchlist | Early interest signal | “Topic worth watching” | High |
| Friends traded similar theme | Social momentum feature | Theme-level article, not individual disclosure | High |
| Cohort trading spike | Crowd momentum article | “Users like you are paying attention” | Medium-High |
| Cohort profit-taking | Timing/urgency signal | “Momentum after recent gains” | High |
| Cohort dip-buying | Contrarian opportunity article | Dip-buyer clusters | High |
| Portfolio similarity group action | Peer benchmark article | “Investors with similar exposure are reviewing this” | High |
| Group chat topic spike | Content trigger | “Conversation starter” article | High |
| Leaderboard movement | Competitive/social article | High-engagement users | Medium-High |
| Referral friend activity | Activation trigger | “A topic to discuss with your network” | High |
| Aggregate theme popularity among friends | FOMO-style topic selection | Theme article without naming friends | High |

### 2.11 Internal and Proprietary Signals

Internal signals can become the main differentiation once public signals commoditize.

| Signal | Use Case | Personalization Angle | Acquisition Difficulty |
|---|---|---|---|
| House view change | Editorial/research trigger | “Our view changed on this name/theme” | Low-Medium |
| Internal conviction score | Ranking and article priority | Higher conviction gets more exposure | Medium |
| Analyst/editor note | Content enrichment | More credible article body | Medium |
| Internal theme basket | Theme recommendation | Users missing or exposed to theme | Medium |
| Model-derived fair value gap | Opportunity article | Valuation-sensitive users | High |
| Proprietary sentiment model | Trigger/ranking feature | Sentiment shift personalization | High |
| User-level conversion propensity | Ranking | Target users likely to act | Medium |
| User-level fatigue score | Suppression | Reduce over-messaging | Medium |
| Cluster response curves | Policy selection | Which cluster responds to which signal | High |
| Article quality score | Content selection | Promote formats that work | Medium |
| Editorial priority | Business alignment | Push strategic themes | Low |
| Revenue proxy per article | Monetization ranking | Prioritize high-value opportunities | Medium |

### 2.12 Derived Composite Signals

Composite signals are often more predictive than raw events.

| Composite Signal | Components | Use Case | Difficulty |
|---|---|---|---|
| Urgency score | price move + news freshness + volatility + market hours | Delivery timing | Medium |
| Personal relevance score | holding/watchlist + position size + past interest | Ranking | Medium |
| Trade readiness score | cash + recent activity + order-ticket history + active hours | Conversion ranking | Medium-High |
| FOMO intensity score | viral attention + cohort activity + price momentum + friend/theme activity | Social urgency framing | High |
| Risk alert score | negative news + exposure + volatility + concentration | Risk article | Medium |
| Opportunity score | positive catalyst + valuation/momentum + user fit | Article prioritization | Medium-High |
| Narrative novelty score | new information vs already-seen content | Avoid stale content | Medium |
| Cluster-policy confidence | historical lift + sample size + model stability | Exploration budget | High |
| Fatigue-adjusted value | expected lift - fatigue/churn cost | Send/suppress decision | Medium |

## 3. Feature Roadmap by Data Availability

The product roadmap should be understood as a data-availability roadmap, not a sequence of different systems. The causal graph inference system exists from day one. At each stage, newly available data unlocks better goals: first relevance, then personalization, then intent prediction, then social momentum, then proprietary persuasion, and eventually multi-step behavior shaping.

As the data improves, the articles improve even if the core architecture remains stable. Better data means the system can choose more relevant signals, write sharper article angles, select stronger CTAs, and publish at more effective moments. This can create new user behavior: users who previously only clicked may start adding to watchlists; users who previously only watched may start opening order tickets; users who previously traded only after obvious price moves may start responding to thematic or peer-driven catalysts.

### Stage 0: Measurement Availability

**Data available:** Article exposures, clicks, dwell time, CTA clicks, asset page views, watchlist actions, order funnel events, executed trades, timestamps, channel metadata, and article metadata.

**Suitable goal:** Make the causal graph observable. The goal is not yet maximum personalization; it is to ensure every article exposure can be connected to subsequent behavior and model decisions can be evaluated later.

**What can be achieved:**

- Canonical event tracking from impression to trade.
- Time-series normalization for market, content, user, portfolio, and outcome events.
- Article metadata schema: signal, ticker, theme, frame, cluster, generation method, send time, channel.
- Attribution windows for immediate, same-day, and 7-day outcomes.
- Baseline dashboards for CTR, click-to-trade, order ticket open, and executed trade.
- Randomized holdout capability by cluster and signal type.
- Day-one causal graph with structural priors and uncertainty estimates.

**Graph capability:** The graph can represent assumed relationships and start learning from interventions. It can answer basic questions such as which article frames get clicks, which signals are associated with trade outcomes, and which channels produce better immediate response.

**User behavior unlocked:** Basic article consumption and early click-to-trade measurement.

### Stage 1: Public Market Data Availability

**Data available:**

- Price movement.
- Volume spikes.
- Basic news.
- Earnings calendar.
- Corporate actions.
- Analyst rating changes if available.
- Macro calendar.

**Suitable goal:** Deliver timely, market-aware articles. The system can create urgency and relevance from public catalysts, but personalization is still coarse unless watchlist or portfolio data is also available.

**What can be achieved:**

- Price move and news-triggered articles.
- Market open, intraday, and close-based urgency logic.
- Ticker, sector, and theme-level article generation.
- Basic headline/frame variants.
- Causal graph edge updates by signal type, frame, timing, and CTR.
- Early trade attribution by event category.

**Example triggers:**

- “A major stock is up/down more than X% today.”
- “A watchlist name is moving after earnings,” if watchlist is available.
- “A sector the user follows is outperforming the market,” if preference data is available.
- “Major macro release is moving growth stocks.”

**Graph capability:** The graph can learn which public signals create attention and which time windows matter. It can optimize article freshness, urgency, headline style, and channel, but it cannot yet fully explain why a signal matters to a specific user.

**User behavior unlocked:** Clicks driven by timeliness, curiosity, and market urgency.

### Stage 2: Watchlist and Basic Preference Availability

**Data available:**

- Watchlists.
- Followed tickers.
- Followed sectors or themes.
- Search history.
- Article topic clicks.
- Preferred reading time and channel response.

**Suitable goal:** Move from market relevance to user-interest relevance. The system can now choose events that match declared or observed interests.

**What can be achieved:**

- Watchlist-triggered articles.
- Search-retargeted articles.
- Topic and ticker preference models.
- Better send-time and channel personalization.
- More accurate suppression of irrelevant public events.
- CTA testing between “read more,” “view asset,” “add alert,” and “compare peers.”

**Example triggers:**

- “A stock you follow is moving after new guidance.”
- “The sector you keep reading about is outperforming today.”
- “A ticker you searched recently has a new catalyst.”

**Graph capability:** The graph can infer which user interests amplify the effect of public signals. It can start separating general market popularity from personal relevance.

**User behavior unlocked:** More repeat clicks, more asset page views, more watchlist actions, and higher return visits.

### Stage 3: Brokerage Portfolio Availability

**Data available:**

- Holdings.
- Recent trades.
- Position size bands.
- Portfolio sector/theme exposure.
- Cash availability band if available.

**Suitable goal:** Convert generic market events into personal portfolio events. Article quality improves because the article can explain why the event matters to the user’s actual holdings and exposures.

**What can be achieved:**

- Portfolio exposure engine.
- Direct and indirect exposure mapping.
- Article personalization inserts.
- Portfolio archetype clustering.
- Basic trade-readiness score.
- Frequency caps by portfolio relevance.

**Example features:**

- Direct holding alert.
- Peer/competitor article for owned stocks.
- Sector concentration trigger.
- “What this event means for your portfolio” article.
- Follow-up article after recent buy/sell.

**Graph capability:** The graph can learn how portfolio context changes the causal effect of an article. For example, direct ownership may increase CTR, but unrealized gains, position size, or recent buying may determine whether the article produces trade activity.

**User behavior unlocked:** Higher click-to-trade conversion, more portfolio review sessions, and more order ticket opens.

### Stage 4: Behavioral Intent Availability

**Data available:**

- Search behavior.
- Asset page views.
- Chart interactions.
- Watchlist changes.
- Alert creation.
- Order ticket opens and abandonments.
- Article read depth and CTA clicks.

**Suitable goal:** Identify when the user is close to action. The system can now distinguish general interest from active trade intent and can select sharper CTAs.

**What can be achieved:**

- Intent scoring by ticker/theme.
- Abandoned-order follow-up articles.
- Retargeting based on asset page views.
- Personalized send-time optimization.
- CTA optimization by user cluster.
- Multi-step journeys: article -> asset page -> watchlist -> alert -> trade.

**Example triggers:**

- “You looked at NVIDIA twice this week; here is what changed today.”
- “You opened an order ticket but did not trade; here are the latest catalysts.”
- “A stock you recently searched is breaking out.”

**Graph capability:** The graph can infer readiness states. It can learn that the same article has different effects depending on whether the user merely owns a stock, recently viewed it, opened an order ticket, or abandoned a trade.

**User behavior unlocked:** More order ticket opens, recovery of abandoned orders, and higher conversion from article click to trading action.

### Stage 5: Experimentation and Intervention Availability

**Data available:**

- Clean exposure logs.
- Holdout groups.
- Treatment variants.
- User cluster histories.
- Signal metadata.
- Outcome events across multiple windows.

**Suitable goal:** Estimate incremental impact rather than raw propensity. This is where the day-one causal graph becomes materially more reliable because it has actual intervention data.

**What can be achieved:**

- Uplift model for click and trade outcomes.
- Lagged time-series predictive models for candidate edge discovery.
- SHAP-based feature importance review for proposed graph edges.
- Automated causal refutation using holdouts, placebo tests, negative controls, and stability checks.
- Weekly graph refresh, promotion, and rollback workflow.
- Treatment/control design by signal type.
- Delayed-send experiments.
- Article frame-level causal effects.
- Counterfactual performance reports.
- Exploration budget by cluster and signal type.

**Example model decisions:**

- For “frequent tech traders,” analyst upgrades may work better as immediate push notifications.
- For “ETF-heavy passive users,” macro-to-portfolio explainers may drive more meaningful action.
- For “recent-loss holders,” risk framing may get clicks but opportunity framing may drive more order tickets.

**Graph capability:** The graph can separate “user was already going to trade” from “article caused incremental trade probability.” It can select signals and article frames based on expected lift, not just correlation.

**User behavior unlocked:** Better trade lift per article impression, less wasted content, and higher-quality article inventory allocation.

### Stage 6: Social, Cohort, and Network Availability

**Data available:**

- Friend graph.
- Consented social signals.
- Group or community activity.
- Aggregated cohort trading behavior.
- Portfolio similarity clusters.
- Theme popularity within a user’s network or cohort.

**Suitable goal:** Add social energy, peer momentum, and FOMO-style curiosity to article selection and framing. At this stage, the system can create articles that feel timely not only because markets moved, but because the user’s relevant social or peer universe is paying attention.

**What can be achieved:**

- Cohort activity spike detection.
- Portfolio-similar-user momentum.
- Friend-circle topic trend detection.
- Theme-level network popularity scoring.
- Social proof article frames.
- “Conversation starter” content surfaces.

**Recommended implementation pattern:** Convert sensitive individual-level signals into aggregated momentum features before they reach article generation. For example, the model can use “theme momentum among similar users” or “attention spike in user’s network” as ranking features without revealing who traded or what they earned.

**Example triggers:**

- “This theme is heating up among investors with similar portfolios.”
- “A stock in your watchlist is drawing unusual attention today.”
- “AI infrastructure names are moving together; here are the ones most relevant to your portfolio.”

**Graph capability:** The graph can estimate when social momentum increases CTR, when it increases trade probability, and which users are sensitive to network-driven cues versus fundamental or portfolio-driven cues.

**User behavior unlocked:** More curiosity clicks, more discussion-driven engagement, more theme exploration, and stronger urgency around crowded topics.

### Stage 7: Proprietary House View and Advanced Research Availability

**Data available:**

- Internal analyst notes.
- House view ratings.
- Proprietary valuation models.
- Internal theme baskets.
- Internal sentiment models.
- Editorial priority signals.

**Suitable goal:** Improve article substance, not just timing. The system can generate articles with a stronger point of view, better explanation, and clearer action logic.

**What can be achieved:**

- House-view-triggered articles.
- View-change alerts.
- Conviction-weighted ranking.
- Personalized research summaries.
- “Why our view changed” content.
- Portfolio gap articles based on internal themes.

**Graph capability:** The graph can learn whether proprietary conviction, valuation gap, or view-change signals add incremental lift beyond public news and price movement.

**User behavior unlocked:** Higher article depth, stronger CTA response, more research-led trading, and stronger retention among users who value interpretation over raw alerts.

### Stage 8: Longitudinal Journey Availability

**Data available:**

- Longitudinal user histories.
- Multi-touch article exposures.
- Long-window trade outcomes.
- Retention/churn outcomes.
- Cross-channel delivery data.
- Exploration results.

**Suitable goal:** Optimize user journeys rather than isolated articles. The system can learn how a sequence of articles, alerts, asset views, and CTAs changes future behavior.

**What can be achieved:**

- Contextual bandits for article frame, headline, channel, and send time.
- Reinforcement learning for multi-step trading journeys.
- Per-cluster exploration budgets.
- Fatigue-aware optimization.
- Incremental revenue/trade lift optimization.
- Automated policy simulation before rollout.

**Graph capability:** The graph can model dynamic user state transitions: dormant to curious, curious to engaged, engaged to trade-ready, trade-ready to trading, and trading to retained. It can decide whether the next best action is education, urgency, comparison, social proof, suppression, or a direct CTA.

**User behavior unlocked:** New behavior patterns can emerge because article quality, timing, and sequencing improve together. Users may begin relying on the app as a daily trading prompt, not only as a news feed.

## 4. Suggested Build Sequence

### Now: First 0-3 Months

Build the foundation that lets the product learn quickly:

- Event tracking and attribution pipeline.
- Public market event ingestion.
- Entity resolution and security master.
- Article candidate schema.
- Day-one causal graph with structural priors.
- Basic ranking by relevance, urgency, and freshness.
- Holdings and watchlist personalization.
- CTR and trade funnel dashboards.
- Initial holdout design.

### Next: 3-6 Months

Turn the system into a personalization engine:

- Portfolio exposure engine.
- User clustering by portfolio and behavior.
- Intent scoring from in-app behavior.
- Headline/frame experimentation.
- Send-time optimization.
- Trade-readiness scoring.
- Cluster-level model reporting.
- Stronger causal graph estimates for top signal categories.

### Later: 6-12 Months

Build defensibility and advanced optimization:

- Contextual bandits.
- Network/cohort momentum features.
- House-view integration.
- Composite FOMO, urgency, relevance, and fatigue scores.
- Multi-touch attribution.
- Policy simulation and automated retraining.

### Long-Term: 12+ Months

Move from article personalization to portfolio action orchestration:

- Reinforcement learning over multi-step journeys.
- Personalized article sequences.
- Cross-user network intelligence.
- Proprietary research and valuation signals.
- Automated thematic campaigns.
- Retention-adjusted trade optimization.

## 5. Prioritized Signal Roadmap

The following order balances acquisition difficulty, expected CTR impact, and trade conversion potential.

| Priority | Signal Family | Why It Comes Here |
|---|---|---|
| 1 | Price moves, volume, news, earnings calendar | Easy to acquire, high timeliness, strong article triggers |
| 2 | Watchlist and holdings exposure | Turns generic events into personal events |
| 3 | Article engagement and asset page behavior | Proprietary, easy to collect, strong intent proxy |
| 4 | Recent trades and portfolio state | Improves conversion prediction and article framing |
| 5 | Earnings surprise, analyst changes, corporate actions | High-quality security-specific catalysts |
| 6 | Sector/theme mapping and peer exposure | Enables indirect personalization and more article inventory |
| 7 | Order funnel behavior | Identifies near-trade users and abandoned intent |
| 8 | Uplift model features and holdout data | Allows causal optimization instead of correlation chasing |
| 9 | Social/viral public attention | High CTR, useful for urgency and curiosity |
| 10 | Cohort and portfolio-similar activity | Proprietary social proof without needing friend-level article claims |
| 11 | Friend/network momentum | Highly differentiated but difficult and sensitive |
| 12 | Internal house view and proprietary models | Strategic differentiation and higher-quality recommendations |
| 13 | Dynamic causal graph and bandit policies | Highest optimization power, but requires large clean data volume |

## 6. Practical Implementation Notes

### 6.1 Data Contracts

Every article should have immutable metadata at exposure time. This is critical because user cluster, portfolio, price, and market state can change later. Store the exact features used for decisioning so offline evaluation can reproduce why the article was sent.

Key metadata should include:

- User ID.
- Cluster ID.
- Article ID.
- Signal event ID.
- Trigger source.
- Ticker/theme IDs.
- Article frame.
- Headline variant.
- Channel.
- Send timestamp.
- Rank score and component scores.
- Model version.
- Policy version.
- Experiment ID.
- Suppression context: which other articles were available but not sent.

### 6.2 Article Frames to Test

The same event should be tested across different frames:

- **Urgency frame**: “Why this move matters now.”
- **Opportunity frame**: “What could change next.”
- **Risk frame**: “What holders should watch.”
- **Peer comparison frame**: “How this compares with similar names.”
- **Portfolio impact frame**: “What this means for your portfolio.”
- **Social proof frame**: “Why this name is getting attention.”
- **House view frame**: “Our take after today’s move.”
- **Contrarian frame**: “The market reaction may be overdone.”
- **Educational frame**: “What this event means in plain English.”
- **Action frame**: “Three things to consider before market close.”

### 6.3 Attribution Windows

Use different attribution windows by signal type:

- Price spike: 0-2 hours and same day.
- Earnings: same day and 1-3 days.
- Analyst change: same day and 1-7 days.
- Macro: same day.
- Thematic article: 3-14 days.
- House view: 7-30 days.
- Social momentum: 0-24 hours.

### 6.4 Exploration Strategy

Exploration should be controlled and cluster-aware. New article frames should first be tested on users with high engagement tolerance and lower fatigue risk. Once a frame shows positive incremental lift, expand it to adjacent clusters.

Useful exploration dimensions:

- Headline framing.
- Send time.
- Channel.
- CTA.
- Article length.
- Chart vs no chart.
- Direct holding vs peer framing.
- Urgency vs explanation.
- Social proof vs fundamental rationale.

### 6.5 Common Failure Modes

The team should monitor for these failure modes:

- Optimizing for users who would have traded anyway.
- Over-sending during volatile markets.
- Confusing asset popularity with article effectiveness.
- Treating CTR as success when trade conversion is poor.
- Treating trade conversion as success when retention worsens.
- Overfitting to active traders and ignoring dormant-user activation.
- Using stale portfolio snapshots.
- Missing market-hour effects.
- Double-counting users exposed to multiple related articles.
- Letting viral signals dominate higher-quality signals.

## 7. Recommended MVP Scope

The MVP should include:

- Public event ingestion for price moves, news, earnings, and corporate actions.
- Portfolio and watchlist matching.
- Article candidate generation from templates.
- Basic personalization inserts.
- Relevance, urgency, freshness, and fatigue scoring.
- Push/in-app delivery.
- Exposure and outcome logging.
- CTR, order ticket open, and executed trade attribution.
- Holdout groups.
- Manual analyst console for reviewing top triggers.

The MVP should include causal graph inference from day one, but the day-one graph should be sparse and uncertainty-aware. It should combine structural priors, deterministic relevance rules, and early outcome learning. As more data becomes available, the same graph should become more granular and more confident rather than being replaced by a separate “advanced” system later.

## 8. North Star System Design Principle

The winning system is not the one with the most signals. It is the one that best answers this question in real time:

“For this user, in this portfolio state, under this market condition, which signal and article frame will create the largest incremental probability of a valuable action?”

Everything in the architecture should support that decision.
