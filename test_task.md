## [cite_start]Test Task: Pump.fun Pre-Buy Scoring & Exit Strategy [cite: 1]

### [cite_start]Objective [cite: 2]
[cite_start]Develop a logic prototype that: [cite: 3]
* [cite_start]Decides within the first **200–300 ms** whether to buy a token on Pump.fun (pre-buy scoring)[cite: 4].
* [cite_start]Proposes rules/models for exiting (selling) based on post-launch activity[cite: 5].

---

### [cite_start]Input Data [cite: 6]

#### [cite_start]1. Data from gRPC at the moment of token creation [cite: 7]
* [cite_start]**deployer_address**: the address of the deployer[cite: 8].
* **token_metadata**: name, ticker, image_hash, description, website, twitter_handle, telegram_link[cite: 9].
* [cite_start]**deployer_deposit_amount**: the amount the deployer's address was topped up by[cite: 10].
* **deployer_wallet_balance_before**: the balance before the last top-up[cite: 11].
* **deployer_wallet_source**: which wallet funded the deployer (mark if it was a CEX)[cite: 12].
* [cite_start]Number of tokens created on the deployer's wallet[cite: 13].
* [cite_start]Number of transactions on the deployer's wallet[cite: 14].
* Any other data that can be obtained in advance[cite: 15].

#### 2. Data for 30–60 minutes post-creation [cite: 16]
* [cite_start]**holders_distribution**: distribution of holders[cite: 17].
* **deployer_actions**: whether they changed the avatar, if there is a tweet, or if the text was edited[cite: 18].
* **volume_curve**: volume by slots/seconds[cite: 19].
* **price_curve**: price curve[cite: 20].
* **top_wallets_activity**: whether strong players are buying[cite: 21].
* [cite_start]**peak_marketcap**: the peak market cap during the period[cite: 22].

#### [cite_start]3. Historical Database [cite: 23]
[cite_start]A database of the last million tokens on Pump.fun with standard fields (including deployer and market cap at the time of the last check)[cite: 24].

---

### [cite_start]Ready Dataset [cite: 25]
[cite_start]The dataset consists of 3 tables: [cite: 26, 27]

| Table | Description | Link |
| :--- | :--- | :--- |
| **tokens** | Main token table | [tokens.parquet](tokens.parquet) |
| **slot_features_60m** | Token behavior in the first 60 minutes post-deploy by slots | [slot_features_60m.parquet](slot_features_60m.parquet) |
| **deployer_actions_60m** | On-chain deployer actions in the first 60 minutes post-deploy | [deployer_actions_60m.parquet](deployer_actions_60m.parquet) |

* [cite_start]**Additional data source**: [https://dune.com/](https://dune.com/)[cite: 29].

---

### [cite_start]Tasks to Complete [cite: 30]

#### [cite_start]Part 1. Instant Decision (Pre-Buy Model) [cite: 31]
1. [cite_start]Assemble a table of ~1000 tokens that have fields from the moment of creation[cite: 32].
2. [cite_start]Build a **scoring model (0–100 points)** that decides whether to buy in the first slot[cite: 33, 34].
3. [cite_start]Define and describe heuristics, for example: [cite: 35]
    * [cite_start]If deployer is funded from a CEX: **+20**[cite: 36].
    * If the image is unique: **+15**[cite: 37].
    * [cite_start]If deployer deposit > 1 SOL: **+20**[cite: 38].
    * [cite_start]If they haven't created tokens before: **+10**[cite: 39].
    * If the name is not similar to the 10 previous ones that day: **+10**[cite: 40].
4. **Goal**: Accuracy in selecting tokens with a potential ROI > 2x after 30 minutes[cite: 41].

#### Part 2. Selling (Exit Strategy Model) [cite: 42]
Analyze 100+ already purchased tokens and: [cite: 43]
1. Build 2–3 exit logics, such as: [cite: 44]
    * [cite_start]Sell at **2x** price[cite: 45].
    * [cite_start]Sell if volume stagnates for **10 consecutive slots**[cite: 46].
    * Sell if a **top address** starts selling[cite: 47].
2. Simulate these strategies on historical data: [cite: 48]
    * [cite_start]What is the average ROI for each? [cite: 49]
    * [cite_start]What is the volatility / maximum drawdown? [cite: 50]
    * How long is the average holding time? [cite: 51]

#### Part 3. Improvement Ideas (Briefly) [cite: 52]
* [cite_start]What can be improved in the future? [cite: 53]
* [cite_start]Can **ML** be used (and at what stage)? [cite: 54]
* What data would you add to the **gRPC stream** to improve pre-buy decisions? [cite: 55]

---

### Example Result [cite: 56]
* [cite_start]A table with 1000 tokens, scoring, and a "buy / don't buy" result[cite: 57].
* [cite_start]1–2 charts showing which features are important[cite: 58].
* Comparison of 2 exit strategies[cite: 59].
* [cite_start]Brief conclusion on how to improve[cite: 60].