import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import websockets
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QLineEdit,
    QPushButton,
    QGroupBox,
    QFormLayout,
    QTableWidget,
    QTableWidgetItem,
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy import stats

# --------------------
# Configure Logging
# --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("simulator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trade_simulator")

# --------------------
# Constants
# --------------------
WEBSOCKET_URL = "wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP"
EXCHANGE = "OKX"
DEFAULT_SYMBOL = "BTC-USDT-SWAP"
DEFAULT_QUANTITY = 100       # USD equivalent
DEFAULT_VOLATILITY = 0.02    # 2%
DEFAULT_FEE_TIER = "VIP0"    # Default OKX fee tier

# Fee structure for OKX (as of documentation)
FEE_TIERS = {
    "VIP0": {"maker": 0.0008, "taker": 0.0010},
    "VIP1": {"maker": 0.0007, "taker": 0.0009},
    "VIP2": {"maker": 0.0006, "taker": 0.0008},
    "VIP3": {"maker": 0.0005, "taker": 0.0007},
    "VIP4": {"maker": 0.0003, "taker": 0.0005},
    "VIP5": {"maker": 0.0000, "taker": 0.0003},
}

MAX_ORDERBOOK_LEVELS = 50
UPDATE_INTERVAL_MS = 100
MAX_HISTORY_SIZE = 300

# --------------------
# OrderBook Class
# --------------------
class OrderBook:
    """
    Maintains a rolling history of L2 orderbook snapshots (asks & bids),
    mid-price, spread, and various volume metrics.
    """
    def __init__(self):
        self.asks: List[Tuple[float, float]] = []
        self.bids: List[Tuple[float, float]] = []
        self.timestamp: Optional[datetime] = None
        self.symbol: str = ""
        self.exchange: str = ""
        self.last_update_time: float = time.time()

        # Historical data (all lists capped to MAX_HISTORY_SIZE)
        self.price_history: List[Tuple[float, float]] = []         # (timestamp, mid_price)
        self.spread_history: List[Tuple[float, float]] = []        # (timestamp, spread)
        self.mid_price_history: List[float] = []                   # just mid_price for volatility
        self.volume_history: List[Tuple[float, float]] = []        # (timestamp, total_near_volume_usd)
        self.ask_volume_history_units: List[Tuple[float, float]] = []  # (timestamp, sum of top-10 ask units)
        self.bid_volume_history_units: List[Tuple[float, float]] = []  # (timestamp, sum of top-10 bid units)

    def update(self, data: Dict):
        """
        Given a new JSON snapshot from the WebSocket,
        parse and store top MAX_ORDERBOOK_LEVELS asks/bids,
        compute mid-price/spread, and update rolling histories.
        """
        self.timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        self.symbol = data["symbol"]
        self.exchange = data["exchange"]

        # Parse and convert to float
        self.asks = [(float(price), float(qty)) for price, qty in data.get("asks", [])[:MAX_ORDERBOOK_LEVELS]]
        self.bids = [(float(price), float(qty)) for price, qty in data.get("bids", [])[:MAX_ORDERBOOK_LEVELS]]

        # Sort asks ascending by price, bids descending by price
        self.asks.sort(key=lambda x: x[0])
        self.bids.sort(key=lambda x: x[0], reverse=True)

        current_time = time.time()
        self.last_update_time = current_time

        if self.asks and self.bids:
            best_ask = self.asks[0][0]
            best_bid = self.bids[0][0]
            mid_price = (best_ask + best_bid) / 2.0
            spread = best_ask - best_bid

            # Store mid-price & spread history
            self.price_history.append((current_time, mid_price))
            self.spread_history.append((current_time, spread))
            self.mid_price_history.append(mid_price)

            # Compute "near" volume in USD within ±0.1% of mid-price
            threshold = mid_price * 0.001
            near_ask_units = sum(qty for price, qty in self.asks if price <= mid_price + threshold)
            near_bid_units = sum(qty for price, qty in self.bids if price >= mid_price - threshold)
            self.volume_history.append((current_time, (near_ask_units + near_bid_units) * mid_price))

            # Top-10 levels volume (in units, not USD)
            top_10_ask_units = sum(qty for price, qty in self.asks[:10])
            top_10_bid_units = sum(qty for price, qty in self.bids[:10])
            self.ask_volume_history_units.append((current_time, top_10_ask_units))
            self.bid_volume_history_units.append((current_time, top_10_bid_units))

            # Trim histories if they exceed MAX_HISTORY_SIZE
            if len(self.price_history) > MAX_HISTORY_SIZE:
                self.price_history = self.price_history[-MAX_HISTORY_SIZE:]
                self.spread_history = self.spread_history[-MAX_HISTORY_SIZE:]
                self.mid_price_history = self.mid_price_history[-MAX_HISTORY_SIZE:]
                self.volume_history = self.volume_history[-MAX_HISTORY_SIZE:]
                self.ask_volume_history_units = self.ask_volume_history_units[-MAX_HISTORY_SIZE:]
                self.bid_volume_history_units = self.bid_volume_history_units[-MAX_HISTORY_SIZE:]

    def get_liquidity_at_level(self, usd_amount: float, side: str = "buy") -> Tuple[float, float]:
        """
        Placeholder for a method that would compute how many units you can buy/sell
        for a given USD amount at current depth, or the price impact to fill that USD amount.
        Not implemented in this example.
        """
        return 0.0, 0.0


# --------------------
# OrderbookProcessor Class
# --------------------
class OrderbookProcessor:
    """
    Takes each L2 snapshot, computes mid-price, spread, top-10 volumes,
    and maintains a small rolling training set for two models:
      1) LinearRegression for slippage (continuous target)
      2) LogisticRegression for maker/taker (binary target)
    """
    def __init__(self):
        # In-memory history (capped at 1000 entries)
        self.orderbook_history: List[Dict] = []
        self.price_history: List[float] = []
        self.spread_history: List[float] = []
        self.volume_history: List[Tuple[float, float]] = []   # (ask_vol, bid_vol)
        self.timestamp_history: List[datetime] = []
        self.processing_times: List[float] = []

        # Regression models
        self.slippage_model = LinearRegression()
        self.maker_taker_model = LogisticRegression()
        self.is_model_trained = False

        # Current state
        self.current_asks: List[List[float]] = []
        self.current_bids: List[List[float]] = []
        self.current_mid_price: float = 0.0
        self.current_spread: float = 0.0

    def process_orderbook(self, data: Dict) -> float:
        """
        Called on each WebSocket message. Parses asks/bids, computes mid-price & spread,
        updates rolling history arrays, and occasionally retrains both models.
        Returns the processing time (in seconds) for latency display.
        """
        start_time = time.time()

        try:
            timestamp = data.get("timestamp")
            asks = data.get("asks", [])
            bids = data.get("bids", [])

            # Convert to floats
            asks = [[float(price), float(size)] for price, size in asks]
            bids = [[float(price), float(size)] for price, size in bids]

            # Sort: asks ascending, bids descending
            asks.sort(key=lambda x: x[0])
            bids.sort(key=lambda x: x[0], reverse=True)

            self.current_asks = asks
            self.current_bids = bids

            if asks and bids:
                best_ask = asks[0][0]
                best_bid = bids[0][0]
                mid_price = (best_ask + best_bid) / 2.0
                spread = best_ask - best_bid

                self.current_mid_price = mid_price
                self.current_spread = spread

                # Top-10 volumes
                ask_vol = sum(size for _, size in asks[:10])
                bid_vol = sum(size for _, size in bids[:10])

                # Store in history
                self.price_history.append(mid_price)
                self.spread_history.append(spread)
                self.volume_history.append((ask_vol, bid_vol))

                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                self.timestamp_history.append(dt)

                # Keep only top-10 levels in memory for training
                snapshot = {
                    "timestamp": dt,
                    "asks": asks[:10],
                    "bids": bids[:10],
                    "mid_price": mid_price,
                    "spread": spread
                }
                self.orderbook_history.append(snapshot)

                # Trim if too large
                if len(self.orderbook_history) > 1000:
                    self.orderbook_history.pop(0)
                    self.price_history.pop(0)
                    self.spread_history.pop(0)
                    self.volume_history.pop(0)
                    self.timestamp_history.pop(0)

                # Train models once we have >100 snapshots
                if len(self.orderbook_history) > 100 and not self.is_model_trained:
                    self._train_models()
                    self.is_model_trained = True
                # Retrain every 100 new snapshots thereafter
                elif len(self.orderbook_history) % 100 == 0 and self.is_model_trained:
                    self._train_models()

            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            return processing_time

        except Exception as e:
            logger.error(f"Error processing orderbook: {e}")
            return time.time() - start_time

    def _train_models(self):
        """
        Build feature matrix X (spread, ask_vol, bid_vol, volatility)
        and two targets:
          y_slippage (continuous) = |(next_mid - current_mid)/current_mid|
          y_maker_taker (binary) = 1 if bid_vol > ask_vol else 0
        Retrain both LinearRegression and LogisticRegression.
        """
        try:
            X: List[List[float]] = []
            y_slippage: List[float] = []
            y_maker_taker: List[int] = []

            # Build features & targets from history length N:
            # We can only define targets for indices 0..N-2 (because we look at next price).
            N = len(self.orderbook_history)
            for i in range(N - 1):
                spread_i = self.spread_history[i]
                ask_vol_i, bid_vol_i = self.volume_history[i]

                # Volatility over the past 20 mid-prices (or fewer if i < 20)
                window_start = max(0, i - 20)
                recent_prices = self.price_history[window_start : i + 1]
                if len(recent_prices) > 1 and np.mean(recent_prices) != 0:
                    vol_i = np.std(recent_prices) / np.mean(recent_prices)
                else:
                    vol_i = 0.0

                X.append([spread_i, ask_vol_i, bid_vol_i, vol_i])

                # Slippage target: |(mid[i+1] - mid[i]) / mid[i]|
                price_i = self.price_history[i]
                price_next = self.price_history[i + 1]
                if price_i != 0:
                    slip = abs((price_next - price_i) / price_i)
                else:
                    slip = 0.0
                y_slippage.append(slip)

                # Maker/Taker target: 1 if bid_vol_i > ask_vol_i else 0
                is_maker = 1 if bid_vol_i > ask_vol_i else 0
                y_maker_taker.append(is_maker)

            if len(X) > 0:
                X_arr = np.array(X)

                # Train the slippage (Linear Regression) model
                self.slippage_model.fit(X_arr, np.array(y_slippage))

                # Train the maker/taker (Logistic Regression) model
                self.maker_taker_model.fit(X_arr, np.array(y_maker_taker))

                logger.info("Models trained successfully")

        except Exception as e:
            logger.error(f"Error training models: {e}")

    def calculate_metrics(self, quantity: float, volatility: float, fee_tier: str) -> Dict[str, float]:
        """
        Given the user’s desired USD quantity, volatility input, and chosen fee tier,
        return a dict containing:
          - slippage (predicted fraction)
          - fees (maker/taker blended)
          - market_impact (Almgren-Chriss)
          - net_cost (slippage + fees + impact) * quantity
          - maker_taker_ratio (string "p_maker/p_taker")
          - latency (avg of last 100 processing times)
        """
        if not self.current_asks or not self.current_bids:
            return {
                "slippage": 0.0,
                "fees": 0.0,
                "market_impact": 0.0,
                "net_cost": 0.0,
                "maker_taker_ratio": "0.00/0.00",
                "latency": 0.0
            }

        try:
            # Build feature vector for current snapshot
            ask_vol = sum(size for _, size in self.current_asks[:10])
            bid_vol = sum(size for _, size in self.current_bids[:10])
            features = np.array([[self.current_spread, ask_vol, bid_vol, volatility]])

            if self.is_model_trained:
                slippage_pred = float(self.slippage_model.predict(features)[0])
                maker_prob = float(self.maker_taker_model.predict_proba(features)[0][1])
            else:
                slippage_pred = 0.001
                maker_prob = 0.5

            taker_prob = 1.0 - maker_prob

            # Compute blended fees
            maker_fee = FEE_TIERS[fee_tier]["maker"]
            taker_fee = FEE_TIERS[fee_tier]["taker"]
            fees = (maker_fee * maker_prob + taker_fee * taker_prob) * quantity

            # Compute market impact
            market_impact = self._calculate_almgren_chriss_impact(quantity, volatility)

            # Net cost
            net_cost = (slippage_pred + market_impact + (fees / quantity)) * quantity

            # Average latency (over last 100)
            if len(self.processing_times) >= 1:
                avg_latency = float(np.mean(self.processing_times[-100:]))
            else:
                avg_latency = 0.0

            return {
                "slippage": slippage_pred,
                "fees": fees,
                "market_impact": market_impact,
                "net_cost": net_cost,
                "maker_taker_ratio": f"{maker_prob:.2f}/{taker_prob:.2f}",
                "latency": avg_latency
            }

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                "slippage": 0.0,
                "fees": 0.0,
                "market_impact": 0.0,
                "net_cost": 0.0,
                "maker_taker_ratio": "0.00/0.00",
                "latency": 0.0
            }

    def _calculate_almgren_chriss_impact(self, quantity: float, volatility: float) -> float:
        """
        Simplified Almgren-Chriss impact:
          MI = σ * sqrt(τ) * (quantity / V) * γ
        where:
          - σ is volatility
          - τ = 1.0 (unit time)
          - V = sum(top-10 ask sizes + top-10 bid sizes)
          - γ = 0.5 (fixed impact factor)
        """
        market_volume = (
            sum(size for _, size in self.current_asks[:10]) +
            sum(size for _, size in self.current_bids[:10])
        )
        if market_volume == 0:
            market_volume = 1.0

        gamma = 0.5
        tau = 1.0
        impact = volatility * np.sqrt(tau) * (quantity / market_volume) * gamma
        return impact


# --------------------
# WebSocketThread Class
# --------------------
class WebSocketThread(QThread):
    """
    QThread that owns its own asyncio event loop and maintains a persistent
    WebSocket connection to stream L2 orderbook data. On each message:
      - Emits data_received(dict) with raw JSON parsed
      - Emits processing_complete(float) with time spent processing
      - Emits connection_status(bool, str) whenever connection state changes
    """
    data_received = pyqtSignal(dict)
    processing_complete = pyqtSignal(float)
    connection_status = pyqtSignal(bool, str)

    def __init__(self, url: str):
        super().__init__()
        self.url = url
        self.running = False
        self.processor = OrderbookProcessor()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    async def connect_and_listen(self):
        """
        Single coroutine that attempts to connect to the WebSocket once,
        then listens for messages, processing each with OrderbookProcessor.
        If a timeout or any exception occurs, it closes the socket, waits
        with exponential backoff, and reconnects—up to a max of 10 retries.
        """
        reconnect_delay = 1
        max_reconnect_delay = 60
        reconnect_attempts = 0
        max_reconnect_attempts = 10

        while self.running and (reconnect_attempts < max_reconnect_attempts):
            reconnect_attempts += 1
            self.connection_status.emit(False, f"Attempting to connect (Attempt {reconnect_attempts})...")
            try:
                # Disable library pings; we handle our own timeout with wait_for(...)
                async with websockets.connect(self.url, ping_interval=None) as websocket:
                    self.connection_status.emit(True, "Connected")
                    logger.info(f"Connected to {self.url}")

                    # Reset backoff counters after a successful connect
                    reconnect_delay = 1
                    reconnect_attempts = 0

                    while self.running:
                        try:
                            # Wait at most 5 seconds for a message
                            message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                            data = json.loads(message)

                            # Emit raw data for UI display
                            self.data_received.emit(data)

                            # Process via OrderbookProcessor
                            processing_time = self.processor.process_orderbook(data)
                            self.processing_complete.emit(processing_time)

                        except asyncio.TimeoutError:
                            # No data for 5 seconds → force reconnect
                            logger.warning("WebSocket recv timeout; reconnecting...")
                            self.connection_status.emit(False, "Timeout, reconnecting...")
                            await websocket.close()
                            break

                        except Exception as ex:
                            logger.error(f"Error during message handling: {ex}")
                            self.connection_status.emit(False, f"Error: {str(ex)}")
                            try:
                                await websocket.close()
                            except:
                                pass
                            break

                    # Exiting inner loop either means: running=False or we hit an exception/timeout.
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
                self.connection_status.emit(False, f"Connection error: {str(e)}")

            if not self.running:
                # If we’ve been told to stop, break out immediately
                break

            if reconnect_attempts >= max_reconnect_attempts:
                logger.error(f"Max reconnection attempts ({max_reconnect_attempts}) reached. Giving up.")
                self.connection_status.emit(False, "Max reconnect attempts reached")
                break

            # Wait with exponential backoff, then retry
            logger.info(f"Waiting {reconnect_delay} seconds before reconnecting...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

        logger.info("Exiting WebSocketThread.connect_and_listen()")

    def run(self):
        """
        Overridden QThread.run()—create an asyncio loop in this thread,
        set it as the current loop, then run connect_and_listen() until stop().
        """
        self.running = True
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.loop.run_until_complete(self.connect_and_listen())
        finally:
            self.loop.close()
            logger.info("WebSocket thread event loop closed.")

    def stop(self):
        """
        Signal the thread to stop. Also stop the asyncio loop if it’s blocked.
        """
        self.running = False
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)

    def get_metrics(self, quantity: float, volatility: float, fee_tier: str) -> Dict[str, float]:
        """
        Proxy to OrderbookProcessor.calculate_metrics(...)
        """
        return self.processor.calculate_metrics(quantity, volatility, fee_tier)


# --------------------
# MainWindow Class
# --------------------
class MainWindow(QMainWindow):
    """
    The main PyQt5 GUI window. Left panel for inputs (exchange, symbol, quantity, etc.),
    right panel for output metrics (slippage, fees, impact, etc.), and a QTableWidget
    below that to show the top-10 levels of the live L2 orderbook in real-time.
    """
    def __init__(self):
        super().__init__()

        # Window setup
        self.setWindowTitle("Trade Simulator")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget & layout
        central_widget = QWidget()
        main_layout = QVBoxLayout()

        # ── Top Row: Input (Left) & Output (Right) ───────────────────────────
        top_row = QHBoxLayout()

        # Input Panel (Left)
        left_panel = QGroupBox("Input Parameters")
        left_layout = QFormLayout()

        self.exchange_combo = QComboBox()
        self.exchange_combo.addItem(EXCHANGE)
        left_layout.addRow("Exchange:", self.exchange_combo)

        self.asset_combo = QComboBox()
        self.asset_combo.addItem(DEFAULT_SYMBOL)
        left_layout.addRow("Spot Asset:", self.asset_combo)

        self.order_type_combo = QComboBox()
        self.order_type_combo.addItem("Market")
        left_layout.addRow("Order Type:", self.order_type_combo)

        self.quantity_input = QLineEdit(str(DEFAULT_QUANTITY))
        left_layout.addRow("Quantity (USD):", self.quantity_input)

        self.volatility_input = QLineEdit(str(DEFAULT_VOLATILITY))
        left_layout.addRow("Volatility:", self.volatility_input)

        self.fee_tier_combo = QComboBox()
        for tier in FEE_TIERS.keys():
            self.fee_tier_combo.addItem(tier)
        left_layout.addRow("Fee Tier:", self.fee_tier_combo)

        self.connection_status = QLabel("Disconnected")
        self.connection_status.setStyleSheet("color: red;")
        left_layout.addRow("Status:", self.connection_status)

        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.toggle_connection)
        left_layout.addRow("", self.connect_button)

        left_panel.setLayout(left_layout)
        top_row.addWidget(left_panel, 1)

        # Output Panel (Right)
        right_panel = QGroupBox("Output Parameters")
        right_layout = QFormLayout()

        self.slippage_label = QLabel("0.0000")
        right_layout.addRow("Expected Slippage:", self.slippage_label)

        self.fees_label = QLabel("0.0000")
        right_layout.addRow("Expected Fees:", self.fees_label)

        self.market_impact_label = QLabel("0.0000")
        right_layout.addRow("Expected Market Impact:", self.market_impact_label)

        self.net_cost_label = QLabel("0.0000")
        right_layout.addRow("Net Cost:", self.net_cost_label)

        self.maker_taker_label = QLabel("0.00/0.00")
        right_layout.addRow("Maker/Taker Proportion:", self.maker_taker_label)

        self.latency_label = QLabel("0.0000 ms")
        right_layout.addRow("Internal Latency:", self.latency_label)

        self.price_label = QLabel("0.0000")
        self.price_label.setFont(QFont("Arial", 14, QFont.Bold))
        right_layout.addRow("Current Price:", self.price_label)

        self.depth_label = QLabel("Asks: 0 | Bids: 0")
        right_layout.addRow("Orderbook Depth:", self.depth_label)

        right_panel.setLayout(right_layout)
        top_row.addWidget(right_panel, 1)

        # Add top row to main layout
        main_layout.addLayout(top_row)

        # ── Below: Top-10 L2 Orderbook Table ─────────────────────────────────
        table_group = QGroupBox("Top 10 Levels of L2 Order Book")
        table_layout = QVBoxLayout()

        self.orderbook_table = QTableWidget(10, 4)
        self.orderbook_table.setHorizontalHeaderLabels(
            ["Ask Price", "Ask Size", "Bid Price", "Bid Size"]
        )
        # Let each column stretch equally
        self.orderbook_table.horizontalHeader().setStretchLastSection(True)
        for col in range(4):
            self.orderbook_table.horizontalHeader().setSectionResizeMode(
                col, self.orderbook_table.horizontalHeader().Stretch
            )

        table_layout.addWidget(self.orderbook_table)
        table_group.setLayout(table_layout)

        main_layout.addWidget(table_group)

        # Set main layout
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # ── WebSocket Thread Setup ───────────────────────────────────────────
        self.ws_thread = WebSocketThread(WEBSOCKET_URL)
        self.ws_thread.data_received.connect(self.update_data)
        self.ws_thread.processing_complete.connect(self.update_latency)
        self.ws_thread.connection_status.connect(self.update_connection_status)

        # Timer for updating metrics every UPDATE_INTERVAL_MS ms
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_metrics)
        self.update_timer.start(UPDATE_INTERVAL_MS)

        # Track connection state
        self.is_connected = False

    def toggle_connection(self):
        """Start or stop the WebSocket thread."""
        if not self.is_connected:
            self.ws_thread.start()
            self.connect_button.setText("Disconnect")
            self.is_connected = True
        else:
            self.ws_thread.stop()
            self.ws_thread.wait()
            self.connect_button.setText("Connect")
            self.is_connected = False
            self.connection_status.setText("Disconnected")
            self.connection_status.setStyleSheet("color: red;")

    def update_connection_status(self, connected: bool, message: str):
        """Update the status label when the WebSocket thread signals a status change."""
        self.connection_status.setText(message)
        if connected:
            self.connection_status.setStyleSheet("color: green;")
        else:
            # Use orange if reconnecting or on error
            self.connection_status.setStyleSheet("color: orange;")

    def update_data(self, data: Dict):
        """
        Called whenever the WebSocket thread emits new L2 data.
        1) Update mid-price display.
        2) Update depth summary (counts).
        3) Populate the 10×4 QTableWidget with top-10 asks/bids.
        """
        try:
            asks = data.get("asks", [])
            bids = data.get("bids", [])

            # 1) If at least one ask & one bid, show mid-price
            if asks and bids:
                best_ask = float(asks[0][0])
                best_bid = float(bids[0][0])
                mid_price = (best_ask + best_bid) / 2.0
                self.price_label.setText(f"{mid_price:.2f}")

                # 2) Depth summary: how many levels were received
                self.depth_label.setText(f"Asks: {len(asks)} | Bids: {len(bids)}")

            # 3) Clear all cells in the 10×4 table
            for row in range(10):
                for col in range(4):
                    self.orderbook_table.setItem(row, col, QTableWidgetItem(""))

            # 4) Fill up to 10 rows of asks (lowest ask price first)
            for i in range(min(10, len(asks))):
                price_i = float(asks[i][0])
                size_i = float(asks[i][1])
                item_price = QTableWidgetItem(f"{price_i:.2f}")
                item_price.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                item_size = QTableWidgetItem(f"{size_i:.6f}")
                item_size.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.orderbook_table.setItem(i, 0, item_price)
                self.orderbook_table.setItem(i, 1, item_size)

            # 5) Fill up to 10 rows of bids (highest bid price first)
            for i in range(min(10, len(bids))):
                price_i = float(bids[i][0])
                size_i = float(bids[i][1])
                item_price = QTableWidgetItem(f"{price_i:.2f}")
                item_price.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                item_size = QTableWidgetItem(f"{size_i:.6f}")
                item_size.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.orderbook_table.setItem(i, 2, item_price)
                self.orderbook_table.setItem(i, 3, item_size)

        except Exception as e:
            logger.error(f"Error updating data in GUI: {e}")

    def update_latency(self, processing_time: float):
        """Show the latest processing latency (in ms)."""
        self.latency_label.setText(f"{processing_time * 1000:.4f} ms")

    def update_metrics(self):
        """
        Every UPDATE_INTERVAL_MS ms, recalculate and display:
          - Expected slippage
          - Expected fees
          - Expected market impact
          - Net cost
          - Maker/Taker ratio
        """
        if not self.is_connected:
            return

        try:
            quantity = float(self.quantity_input.text())
            volatility = float(self.volatility_input.text())
            fee_tier = self.fee_tier_combo.currentText()

            metrics = self.ws_thread.get_metrics(quantity, volatility, fee_tier)
            self.slippage_label.setText(f"{metrics['slippage']:.6f}")
            self.fees_label.setText(f"{metrics['fees']:.6f}")
            self.market_impact_label.setText(f"{metrics['market_impact']:.6f}")
            self.net_cost_label.setText(f"{metrics['net_cost']:.6f}")
            self.maker_taker_label.setText(metrics["maker_taker_ratio"])
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")


# --------------------
# Application Entry Point
# --------------------
def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()

