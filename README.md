# GoQuant-Assignment2

# Real-Time Crypto Trade Simulator (OKX)

## 🚀 Objective

To create a high-performance trade simulator that leverages real-time L2 order book data from cryptocurrency exchanges (initially OKX) to estimate transaction costs and market impact for spot assets. The system will connect to WebSocket endpoints, process data in real-time, and present results through a user interface.

**🚨 Important:** Accessing OKX market data may require a VPN depending on your geographical location. Public market data does not require an account.

## ✨ Features

*   **Real-time Data Processing:** Connects to OKX WebSocket for L2 order book data.
*   **Dynamic UI:**
    *   Left Panel: User inputs for simulation parameters.
    *   Right Panel: Real-time display of processed output values.
*   **Comprehensive Cost Estimation:**
    *   Expected Slippage (Regression-based)
    *   Expected Fees (Rule-based)
    *   Expected Market Impact (Almgren-Chriss model adaptation)
    *   Net Cost Calculation
*   **Trade Flow Prediction:** Maker/Taker proportion (Logistic Regression - though for market orders, it's 100% Taker).
*   **Performance Monitoring:** Internal latency measurement.
*   **Extensible Architecture:** Designed for maintainability and future enhancements.

## 🛠️ Technical Stack (Proposed)

*   **Language:** Python 3.9+
*   **UI:** `tkinter` (standard library, simple) or `PyQt5`/`PySide2` (more feature-rich)
*   **WebSocket Communication:** `websockets` library
*   **Numerical Computation & Data Handling:** `numpy`, `pandas`
*   **Machine Learning:** `scikit-learn`
*   **Asynchronous Programming:** `asyncio`
*   **Logging:** `logging` module
*   **Data Structures:** `collections.deque`, `sortedcontainers` (for efficient order book management, optional)

## ⚙️ Initial Setup

1.  **Review OKX API Documentation:**
    *   Spot WebSocket API: [OKX WebSocket API Docs](https://www.okx.com/docs-v5/en/#websocket-api-public-channel-order-book-channel) (Focus on `books` channel for full L2 depth).
2.  **Set up Development Environment:**
    *   Install Python 3.9+
    *   Create a virtual environment:
        ```bash
        python -m venv venv
        source venv/bin/activate # Linux/macOS
        # venv\Scripts\activate # Windows
        ```
    *   Install core dependencies:
        ```bash
        pip install websockets numpy pandas scikit-learn
        # pip install PyQt5 # Or other UI library if chosen
        ```
3.  **VPN (If Required):** Ensure you have a VPN configured if OKX access is restricted in your region.

## Core Components

### 📊 UI Components

The UI will be divided into two main panels:

1.  **Left Panel: Input Parameters**
    *   Dynamically populated fields for user input.
2.  **Right Panel: Processed Output Values**
    *   Real-time display of calculated metrics, updating with each tick.

### 📥 Input Parameters

1.  **Exchange:**
    *   Type: Dropdown / Pre-filled
    *   Value: `OKX` (initially fixed)
2.  **Spot Asset:**
    *   Type: Text Input / Dropdown (populated from available symbols on connection)
    *   Example: `BTC-USDT` (Note: The provided endpoint is `BTC-USDT-SWAP`, which is a derivative. For SPOT, it would be like `BTC-USDT`)
3.  **Order Type:**
    *   Type: Dropdown / Pre-filled
    *   Value: `market` (initially fixed)
4.  **Quantity:**
    *   Type: Number Input
    *   Unit: In quote currency (e.g., USD equivalent for a BTC-USDT pair).
    *   Example: `100` (for 100 USDT worth of BTC)
5.  **Volatility:**
    *   Type: Number Input (Annualized, e.g., 0.6 for 60%)
    *   Note: This is a key parameter for Almgren-Chriss. Can be user-defined or ideally fetched/estimated from recent market data (e.g., standard deviation of log returns).
6.  **Fee Tier:**
    *   Type: Dropdown / Text Input
    *   Basis: Based on OKX fee schedule (e.g., "VIP 0", "Regular User Tier 1"). This will map to specific taker/maker fee rates.
    *   Reference: [OKX Fee Rates](https://www.okx.com/fees)

### 📤 Output Parameters

1.  **Expected Slippage:**
    *   Model: Linear or Quantile Regression.
    *   Calculation: Predicts the difference between the expected fill price (e.g., mid-price or best bid/ask at decision time) and the actual Volume-Weighted Average Price (VWAP) of execution, based on order size, current book depth/spread, and volatility.
2.  **Expected Fees:**
    *   Model: Rule-based.
    *   Calculation: `Quantity * VWAP_Price * TakerFeeRate`. `TakerFeeRate` is determined by the selected Fee Tier.
3.  **Expected Market Impact (Almgren-Chriss Adaptation):**
    *   Model: Based on Almgren-Chriss principles, simplified for a single immediate order.
    *   Calculation: Estimates the additional cost incurred due to the order's size moving the market price. See "Model Implementation" for details.
4.  **Net Cost:**
    *   Calculation: `|Expected Slippage| + Expected Fees + |Expected Market Impact|`. Absolute values for costs.
5.  **Maker/Taker Proportion:**
    *   Model: For `market` orders, this is always 100% Taker.
    *   Output: `Taker: 100%, Maker: 0%`.
    *   Note: A logistic regression model would be relevant for predicting outcomes of *limit orders*.
6.  **Internal Latency:**
    *   Measurement: Time taken from receiving a WebSocket message to completing all calculations and updating the UI for that tick. Measured in milliseconds.

### 🔌 WebSocket Implementation

1.  **Endpoint:** `wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP`
    *   **Note:** This endpoint provides SWAP data. For SPOT, the channel name and potentially the base URL might differ slightly according to OKX documentation (e.g., `wss://ws.okx.com:8443/ws/v5/public` and subscribing to `books` channel for `BTC-USDT`). We will use the provided endpoint for this exercise.
2.  **Subscription Message (Example for official OKX API):**
    ```json
    {
      "op": "subscribe",
      "args": [
        {
          "channel": "books", // For full order book
          "instId": "BTC-USDT" // For SPOT
        }
      ]
    }
    ```
    *   The provided `gomarket-cpp` endpoint seems to stream directly without an explicit subscription message after connection.
3.  **Sample Response Format (from gomarket-cpp):**
    ```json
    {
      "timestamp": "2025-05-04T10:39:13Z",
      "exchange": "OKX",
      "symbol": "BTC-USDT-SWAP",
      "asks": [ /* ["price_str", "quantity_str"], ... */ ],
      "bids": [ /* ["price_str", "quantity_str"], ... */ ]
    }
    ```
4.  **Data Processing:**
    *   Parse incoming JSON messages.
    *   Convert price and quantity strings to appropriate numerical types (e.g., `Decimal` for precision or `float`).
    *   Maintain an internal representation of the L2 order book (bids sorted descending, asks sorted ascending by price).
    *   For each update ("tick"):
        *   Update the order book.
        *   Recalculate all output parameters based on the current book and input parameters.
        *   Update the UI.

## 🛠️ Model Implementation Details

### 1. Expected Slippage (Regression Model)

*   **Concept:** Slippage is the difference between the expected price of a trade and the price at which the trade is actually executed. For a market order, this is influenced by the bid-ask spread and the depth of the order book consumed.
*   **Model Choice:**
    *   **Linear Regression:** `SlippageAmount = β₀ + β₁*OrderSize_USD + β₂*Spread + β₃*Volatility + β₄*Depth_Level1`
    *   **Quantile Regression:** Useful if slippage distribution is non-normal or has heavy tails. Predicts a certain quantile of slippage.
*   **Features:**
    *   `OrderSize_USD`: The quantity from input parameters.
    *   `Spread`: `BestAsk - BestBid`.
    *   `Volatility`: User input or calculated (e.g., 5-min price volatility).
    *   `Depth_Level1`: Quantity available at the best bid/ask.
    *   (Optional) `Depth_N_Levels`: Sum of quantities over first N levels.
*   **Data Collection (for training, if implementing from scratch):**
    *   Requires historical L2 snapshots and executed trade details. For this simulator, we might start with a pre-trained model or a simplified heuristic based on book traversal.
*   **Simplified Approach for Real-time Estimation (without pre-trained model):**
    *   Simulate "walking the book": For a buy order of `X` USD, calculate the VWAP by consuming ask levels until the order is filled.
    *   `Slippage = VWAP_execution - BestAsk_at_decision_time`. (Can also be `VWAP - MidPrice_at_decision_time`).

### 2. Expected Fees (Rule-Based Model)

*   **Concept:** Exchange fees are typically a percentage of the trade value and differ for makers and takers.
*   **Model:**
    1.  Determine the `TakerFeeRate` based on the user-selected "Fee Tier". This requires a mapping (e.g., a dictionary) of fee tiers to rates from OKX documentation.
    2.  Estimate the execution value: `ExecutedValue = Quantity_USD_equivalent`. (For a crypto-crypto pair, `Quantity_Base * VWAP_Price`).
    3.  `ExpectedFees = ExecutedValue * TakerFeeRate`.
    *   **Note:** For market orders, the fee is always the Taker fee.

### 3. Expected Market Impact (Almgren-Chriss Adaptation)

*   **Reference:** [Understanding Almgren-Chriss Model](https://www.linkedin.com/pulse/understanding-almgren-chriss-model-optimal-portfolio-execution-pal-pmeqc/)
*   **Concept (Almgren-Chriss):** Balances execution speed (market impact) against timing risk (volatility). The model is primarily for *scheduling* an order over time.
*   **Adaptation for a Single, Immediate Market Order:**
    *   We are interested in the *temporary market impact* cost.
    *   A common formulation for temporary impact `h(v)` from trading at rate `v`: `h(v) = ε * sgn(v) + η * |v|^α`
        *   `ε * sgn(v)`: Fixed costs like spread.
        *   `η * |v|^α`: Impact from consuming liquidity. `η` (eta) is a market impact parameter.
    *   Simplified for one trade: `MarketImpactCost = γ * σ * (Q / ADV)^δ`
        *   `Q`: Order quantity (in base currency).
        *   `σ`: Volatility of the asset (daily or intraday).
        *   `ADV`: Average Daily Volume (in base currency). This needs to be fetched or estimated.
        *   `γ` (gamma), `δ` (delta): Market-specific impact parameters (e.g., `δ` often ~0.5-1.0). These can be calibrated or taken from literature.
    *   **Alternative L2-based "Impact":** The cost difference between filling at the best price vs. the VWAP achieved by walking the book.
        *   `ImpactCost = (VWAP_execution - BestPrice_opposite_side) * Quantity_executed_base`
        *   This definition overlaps significantly with "Slippage" when slippage is calculated against `BestPrice_opposite_side`.
    *   **Distinction for this simulator:**
        *   **Slippage (from regression):** Overall predicted deviation from mid/best, statistically derived.
        *   **Market Impact (A-C inspired):** Theoretical additional cost due to order size relative to market conditions (volatility, ADV), using an A-C-like formula. The challenge is parameterizing `γ`, `η`, `δ` without extensive calibration.
        *   We can use `σ` (from input), estimate `ADV` (e.g., from 24h volume if available via API, or a placeholder), and use typical values for `γ` and `δ` (e.g. `γ=0.314`, `δ=0.5` are sometimes cited).
    *   **Practical Calculation for UI:**
        1.  Convert `Quantity_USD` to `Quantity_Base` (e.g., BTC) using current mid-price.
        2.  `ADV`: Assume a placeholder (e.g., 10,000 BTC for BTC-USDT) or attempt to fetch via a REST API call if available.
        3.  `Volatility`: User input.
        4.  `MarketImpactCost_per_unit = γ * Volatility * (Quantity_Base / ADV)^δ`
        5.  `TotalMarketImpactCost = MarketImpactCost_per_unit * Quantity_Base` (This is a price deviation, so multiply by quantity to get cost).

### 4. Net Cost

*   `NetCost = |SlippageCost| + FeesCost + |MarketImpactCost|`
*   Slippage and Market Impact are costs, so their absolute values are added.
    *   `SlippageCost = PredictedSlippage_USD_per_unit * Quantity_Base` (if slippage is per unit) or just the direct output from the regression if it predicts total slippage value.
    *   `FeesCost` from fee model.
    *   `MarketImpactCost` from A-C model.

### 5. Maker/Taker Proportion (Logistic Regression)

*   **For Market Orders:** As stated, it's 100% Taker.
    `P(Taker) = 1.0`, `P(Maker) = 0.0`.
*   **For Future Limit Order Functionality (Model Sketch):**
    *   **Target Variable:** Binary (1 if order was a maker, 0 if taker).
    *   **Features:**
        *   `DistanceToMid`: `(LimitPrice - MidPrice) / MidPrice`. Positive for asks above mid, negative for bids below mid.
        *   `OrderSizeRatio`: `OrderSize / AvgTradeSize` or `OrderSize / DepthAtTouch`.
        *   `Spread`.
        *   `Volatility`.
    *   **Model:** `P(Maker) = 1 / (1 + exp(-(β₀ + β₁*DistanceToMid + ...)))`
    *   **Data Collection:** Requires historical data of submitted limit orders and their execution status (maker/taker fill).

### 6. Internal Latency

*   `start_time = time.perf_counter()` at the beginning of WebSocket message processing.
*   `end_time = time.perf_counter()` after all calculations and UI update calls for that tick.
*   `Latency_ms = (end_time - start_time) * 1000`.

## ⚡ Performance Analysis and Optimization (Bonus)

### Latency Benchmarking

1.  **Data Processing Latency:** Time from WebSocket message receipt to having the order book and relevant features (spread, depth) updated internally.
2.  **Model Calculation Latency:** Time taken by each model (Slippage, Impact, Fees) to compute its output.
3.  **UI Update Latency:** Time taken to render changes in the UI. This can be harder to measure precisely with `tkinter` but can be estimated by timing the calls that update UI elements.
4.  **End-to-End Simulation Loop Latency:** Same as "Internal Latency" defined above.
    *   All metrics should be logged, and statistics like average, 95th percentile, and max should be reported.

### Optimization Techniques

1.  **Memory Management:**
    *   **Efficient Order Book:** Use `collections.deque` with a `maxlen` if only top N levels are needed, or `sortedcontainers.SortedDict` for fully sorted books allowing efficient updates and lookups. Python lists of tuples `(price, quantity)` and sorting on each update can be slow for very high frequency. For L2 `snapshot` + `update` style messages (common in many exchanges), one needs to merge efficiently. The provided `gomarket-cpp` endpoint sends full books, simplifying this but potentially sending more data.
    *   **Numeric Types:** Use `numpy` arrays for calculations where possible for vectorized operations. Be mindful of `float` vs `Decimal` (precision vs. speed). For financial data, `Decimal` is often preferred for prices, but `float` may be acceptable for intermediate calculations if performance is critical and precision loss is managed.
    *   **Object Reuse:** Avoid creating many small objects in tight loops.
2.  **Network Communication:**
    *   **AsyncIO:** Use `async for message in websocket:` for non-blocking WebSocket handling.
    *   **Message Parsing:** `orjson` is faster than standard `json` library if parsing becomes a bottleneck.
3.  **Data Structure Selection:**
    *   **Order Book:** As above, `SortedDict` or carefully managed `list`s/`dict`s. Bids (price descending) and Asks (price ascending).
        *   Example: `bids = SortedDict()` (maps price to quantity), `asks = SortedDict()`.
    *   **Time Series Data:** `pandas.Series` or `numpy.ndarray` for price history if used for volatility calculation.
4.  **Thread Management / Asynchronous Operations:**
    *   **Primary `asyncio` Event Loop:** For WebSocket I/O and UI events (if UI library supports it, e.g., `quamash` for PyQt with asyncio).
    *   **CPU-bound Tasks:** If model calculations are heavy, use `loop.run_in_executor` to run them in a thread pool and not block the `asyncio` loop.
    *   **UI Updates:** Ensure UI updates are done from the main thread. `tkinter` requires this. If using `asyncio` with threads, use a queue or `loop.call_soon_threadsafe` to pass results back to the main thread for UI update.
5.  **Regression Model Efficiency:**
    *   **`scikit-learn`:** Already highly optimized as it uses `numpy` and Cython.
    *   **Feature Engineering:** Pre-compute features efficiently. Avoid redundant calculations per tick.
    *   **Model Simplification:** If complex models are too slow, consider simpler heuristics or models with fewer features.
    *   **Prediction Time:** Focus on minimizing `model.predict()` time.

## 📚 Documentation Requirements (This Document)

This Markdown document serves as the primary design and model documentation.

*   **Model Selection and Parameters:** Detailed in the "Model Implementation Details" section.
*   **Regression Techniques Chosen:** Linear/Quantile Regression for slippage, Logistic Regression for Maker/Taker. Rationale provided.
*   **Market Impact Calculation Methodology:** Almgren-Chriss adaptation explained.
*   **Performance Optimization Approaches:** Listed in the "Performance Analysis and Optimization" section.

