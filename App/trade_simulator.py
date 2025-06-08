import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import websockets
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QComboBox, QLineEdit, 
                            QPushButton, QGridLayout, QGroupBox, QFormLayout)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject, QThread
from PyQt5.QtGui import QFont
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy import stats
import PySimpleGUI as sg
from PySimpleGUI import Window

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simulator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trade_simulator")

# Constants
WEBSOCKET_URL = "wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP"
EXCHANGE = "OKX"
DEFAULT_SYMBOL = "BTC-USDT-SWAP"
DEFAULT_QUANTITY = 100  # USD equivalent
DEFAULT_VOLATILITY = 0.02  # 2%
DEFAULT_FEE_TIER = "VIP0"  # Default fee tier

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

class OrderBook:
    def __init__(self):
        self.asks = []
        self.bids = []
        self.timestamp = None
        self.symbol = ""
        self.exchange = ""
        self.last_update_time = time.time()
        self.price_history = [] # Stores (timestamp, mid_price)
        self.spread_history = [] # Stores (timestamp, spread)
        self.mid_price_history = [] # Stores just mid_price for volatility
        self.volume_history = [] # Stores (timestamp, total_near_volume_usd)
        self.ask_volume_history_units = [] # Stores (timestamp, sum of top 10 ask quantities)
        self.bid_volume_history_units = [] # Stores (timestamp, sum of top 10 bid quantities)
        
    def update(self, data: Dict):
        self.timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        self.symbol = data["symbol"]
        self.exchange = data["exchange"]
        
        # Process and sort asks and bids
        self.asks = [(float(price), float(qty)) for price, qty in data.get("asks", [])[:MAX_ORDERBOOK_LEVELS]]
        self.bids = [(float(price), float(qty)) for price, qty in data.get("bids", [])[:MAX_ORDERBOOK_LEVELS]]
        
        self.asks.sort(key=lambda x: x[0])
        self.bids.sort(key=lambda x: x[0], reverse=True)
        
        current_time = time.time()
        self.last_update_time = current_time
        
        if self.asks and self.bids:
            best_ask = self.asks[0][0]
            best_bid = self.bids[0][0]
            mid_price = (best_ask + best_bid) / 2
            spread = best_ask - best_bid
            
            # Append to history lists
            self.price_history.append((current_time, mid_price))
            self.spread_history.append((current_time, spread))
            self.mid_price_history.append(mid_price) # Keep a list of just prices for volatility
            
            # Calculate and store near volume in USD (using 0.1% threshold around mid price)
            threshold = mid_price * 0.001
            near_ask_volume_units = sum(qty for price, qty in self.asks if price <= mid_price + threshold)
            near_bid_volume_units = sum(qty for price, qty in self.bids if price >= mid_price - threshold)
            self.volume_history.append((current_time, (near_ask_volume_units + near_bid_volume_units) * mid_price)) # Store total near volume in USD

            # Calculate and store total quantity in units for top 10 levels (for model features)
            top_10_ask_volume_units = sum(qty for price, qty in self.asks[:10])
            top_10_bid_volume_units = sum(qty for price, qty in self.bids[:10])
            self.ask_volume_history_units.append((current_time, top_10_ask_volume_units))
            self.bid_volume_history_units.append((current_time, top_10_bid_volume_units))

            # Trim history lists to MAX_HISTORY_SIZE
            if len(self.price_history) > MAX_HISTORY_SIZE:
                self.price_history = self.price_history[-MAX_HISTORY_SIZE:]
                self.spread_history = self.spread_history[-MAX_HISTORY_SIZE:]
                self.mid_price_history = self.mid_price_history[-MAX_HISTORY_SIZE:]
                self.volume_history = self.volume_history[-MAX_HISTORY_SIZE:]
                self.ask_volume_history_units = self.ask_volume_history_units[-MAX_HISTORY_SIZE:]
                self.bid_volume_history_units = self.bid_volume_history_units[-MAX_HISTORY_SIZE:]
    
    def get_liquidity_at_level(self, usd_amount: float, side: str = "buy") -> Tuple[float, float]:
        # Implementation of get_liquidity_at_level method
        pass

class OrderbookProcessor:
    """Processes orderbook data and calculates trading metrics"""
    
    def __init__(self):
        self.orderbook_history = []
        self.price_history = []
        self.spread_history = []
        self.volume_history = []
        self.timestamp_history = []
        self.processing_times = []
        
        # Models
        self.slippage_model = LinearRegression()
        self.maker_taker_model = LogisticRegression()
        self.is_model_trained = False
        
        # Current orderbook state
        self.current_asks = []
        self.current_bids = []
        self.current_mid_price = 0
        self.current_spread = 0
        
    def process_orderbook(self, data: dict) -> float:
        """Process a single orderbook update and return processing time"""
        start_time = time.time()
        
        try:
            # Extract data
            timestamp = data.get("timestamp")
            asks = data.get("asks", [])
            bids = data.get("bids", [])
            
            # Convert to numeric values
            asks = [[float(price), float(size)] for price, size in asks]
            bids = [[float(price), float(size)] for price, size in bids]
            
            # Sort orderbook (asks ascending, bids descending)
            asks.sort(key=lambda x: x[0])
            bids.sort(key=lambda x: x[0], reverse=True)
            
            # Store current state
            self.current_asks = asks
            self.current_bids = bids
            
            # Calculate mid price and spread
            best_ask = asks[0][0] if asks else None
            best_bid = bids[0][0] if bids else None
            
            if best_ask and best_bid:
                mid_price = (best_ask + best_bid) / 2
                spread = best_ask - best_bid
                
                self.current_mid_price = mid_price
                self.current_spread = spread
                
                # Store historical data
                self.price_history.append(mid_price)
                self.spread_history.append(spread)
                
                # Calculate total volume
                ask_volume = sum(size for _, size in asks[:10])  # Top 10 levels
                bid_volume = sum(size for _, size in bids[:10])  # Top 10 levels
                self.volume_history.append((ask_volume, bid_volume))
                
                # Store timestamp
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                self.timestamp_history.append(dt)
                
                # Store full orderbook (limited to top 10 levels for memory efficiency)
                self.orderbook_history.append({
                    'timestamp': dt,
                    'asks': asks[:10],
                    'bids': bids[:10],
                    'mid_price': mid_price,
                    'spread': spread
                })
                
                # Limit history size to prevent memory issues
                if len(self.orderbook_history) > 1000:
                    self.orderbook_history.pop(0)
                    self.price_history.pop(0)
                    self.spread_history.pop(0)
                    self.volume_history.pop(0)
                    self.timestamp_history.pop(0)
                
                # Train models if we have enough data
                if len(self.orderbook_history) > 100 and not self.is_model_trained:
                    self._train_models()
                    self.is_model_trained = True
                # Retrain periodically
                elif len(self.orderbook_history) % 100 == 0 and self.is_model_trained:
                    self._train_models()
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            return processing_time
            
        except Exception as e:
            logger.error(f"Error processing orderbook: {e}")
            return time.time() - start_time
    
    def _train_models(self):
        """Train regression models for slippage and maker/taker prediction"""
        try:
            # Prepare features for slippage model
            X = []
            for i in range(len(self.orderbook_history) - 1):
                features = [
                    self.spread_history[i],
                    self.volume_history[i][0],  # ask volume
                    self.volume_history[i][1],  # bid volume
                    # Volatility estimate (using price changes)
                    np.std(self.price_history[max(0, i-20):i+1]) / np.mean(self.price_history[max(0, i-20):i+1]) if i > 0 else 0
                ]
                X.append(features)
            
            if len(X) > 0:
                X = np.array(X)
                
                # Target for slippage model (using price changes as proxy)
                y_slippage = []
                for i in range(len(self.orderbook_history) - 1):
                    # Simulate slippage as the difference between expected execution price and actual price
                    # For simplicity, we use the next mid price as a proxy for execution price
                    expected_price = self.price_history[i]
                    actual_price = self.price_history[i+1]
                    slippage = abs((actual_price - expected_price) / expected_price)
                    y_slippage.append(slippage)
                
                # Train slippage model
                self.slippage_model.fit(X, y_slippage)
                
                # Target for maker/taker model (using volume imbalance as proxy)
                y_maker_taker = []
                for i in range(len(self.orderbook_history) - 1):
                    ask_vol = self.volume_history[i][0]
                    bid_vol = self.volume_history[i][1]
                    # If bid volume > ask volume, more likely to be executed as maker
                    is_maker = 1 if bid_vol > ask_vol else 0
                    y_maker_taker.append(is_maker)
                
                # Train maker/taker model
                self.maker_taker_model.fit(X, y_maker_taker)
                
                logger.info("Models trained successfully")
        
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
    def calculate_metrics(self, quantity: float, volatility: float, fee_tier: str) -> dict:
        """Calculate trading metrics based on current orderbook and parameters"""
        if not self.current_asks or not self.current_bids:
            return {
                "slippage": 0,
                "fees": 0,
                "market_impact": 0,
                "net_cost": 0,
                "maker_taker_ratio": 0,
                "latency": 0
            }
        
        try:
            # Extract features for prediction
            features = [
                self.current_spread,
                sum(size for _, size in self.current_asks[:10]),  # ask volume
                sum(size for _, size in self.current_bids[:10]),  # bid volume
                volatility
            ]
            features = np.array([features])
            
            # Predict slippage
            slippage = self.slippage_model.predict(features)[0] if self.is_model_trained else 0.001
            
            # Predict maker/taker proportion
            maker_prob = self.maker_taker_model.predict_proba(features)[0][1] if self.is_model_trained else 0.5
            taker_prob = 1 - maker_prob
            
            # Calculate fees based on maker/taker proportion
            maker_fee = FEE_TIERS[fee_tier]["maker"]
            taker_fee = FEE_TIERS[fee_tier]["taker"]
            fees = (maker_fee * maker_prob + taker_fee * taker_prob) * quantity
            
            # Calculate market impact using Almgren-Chriss model
            market_impact = self._calculate_almgren_chriss_impact(quantity, volatility)
            
            # Calculate net cost
            net_cost = (slippage + fees + market_impact) * quantity
            
            # Calculate average processing latency
            avg_latency = np.mean(self.processing_times[-100:]) if self.processing_times else 0
            
            return {
                "slippage": slippage,
                "fees": fees,
                "market_impact": market_impact,
                "net_cost": net_cost,
                "maker_taker_ratio": f"{maker_prob:.2f}/{taker_prob:.2f}",
                "latency": avg_latency
            }
        
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                "slippage": 0,
                "fees": 0,
                "market_impact": 0,
                "net_cost": 0,
                "maker_taker_ratio": 0,
                "latency": 0
            }
    
    def _calculate_almgren_chriss_impact(self, quantity: float, volatility: float) -> float:
        """
        Calculate market impact using Almgren-Chriss model
        
        The Almgren-Chriss model estimates market impact as:
        MI = σ * sqrt(τ) * (quantity / V) * γ
        
        Where:
        - σ is the volatility
        - τ is the time horizon (we use 1 for simplicity)
        - quantity is the order size
        - V is the market volume
        - γ is a market impact factor (typically 0.1-1.0)
        """
        # Market volume (sum of bid and ask volumes)
        market_volume = sum(size for _, size in self.current_asks[:10]) + sum(size for _, size in self.current_bids[:10])
        
        # Avoid division by zero
        if market_volume == 0:
            market_volume = 1
        
        # Market impact factor (can be calibrated based on historical data)
        gamma = 0.5
        
        # Time horizon (using 1 for simplicity)
        tau = 1.0
        
        # Calculate market impact
        impact = volatility * np.sqrt(tau) * (quantity / market_volume) * gamma
        
        return impact

class WebSocketThread(QThread):
    """
    Thread for handling WebSocket connection and data processing,
    rewritten to use a single asyncio event loop in this thread.
    """
    data_received = pyqtSignal(dict)
    processing_complete = pyqtSignal(float)
    connection_status = pyqtSignal(bool, str)
    
    def __init__(self, url):
        super().__init__()
        self.url = url
        self.running = False
        self.processor = OrderbookProcessor()
        self.loop = None  # Will hold the asyncio event loop for this thread
    
    async def connect_and_listen(self):
        """
        This coroutine attempts to connect to the WebSocket, processes
        incoming messages, and if disconnected (timeout/exception), it will
        wait with exponential backoff and retry—up to a maximum number of attempts.
        """
        reconnect_delay = 1            # start with 1 second
        max_reconnect_delay = 60       # cap at 60 seconds
        reconnect_attempts = 0
        max_reconnect_attempts = 10    # give up after 10 full cycles
        
        while self.running and (reconnect_attempts < max_reconnect_attempts):
            reconnect_attempts += 1
            self.connection_status.emit(False, f"Attempting to connect (Attempt {reconnect_attempts})...")
            try:
                # Disable built-in keepalive so that we control timeouts ourselves
                async with websockets.connect(self.url, ping_interval=None) as websocket:
                    # Once connected, reset backoff counters
                    self.connection_status.emit(True, "Connected")
                    logger.info(f"Connected to {self.url}")
                    reconnect_delay = 1
                    reconnect_attempts = 0
                    
                    while self.running:
                        try:
                            # Wait at most 5 seconds for a message:
                            message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                            data = json.loads(message)
                            
                            # Emit raw data for UI
                            self.data_received.emit(data)
                            
                            # Process data
                            processing_time = self.processor.process_orderbook(data)
                            self.processing_complete.emit(processing_time)
                        
                        except asyncio.TimeoutError:
                            # No message received in 5 seconds → force a reconnect
                            logger.warning("WebSocket timeout, reconnecting...")
                            self.connection_status.emit(False, "Timeout, reconnecting...")
                            await websocket.close()
                            break
                        
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            self.connection_status.emit(False, f"Error: {str(e)}")
                            try:
                                await websocket.close()
                            except:
                                pass
                            break
                        
                    # Exiting inner loop means either stopped or we hit an exception/timeout
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
                self.connection_status.emit(False, f"Connection error: {str(e)}")
            
            # If we reach here, it means connection was lost or we encountered an exception.
            if not self.running:
                break
            
            # If we still want to keep trying, wait before reconnecting
            if reconnect_attempts >= max_reconnect_attempts:
                logger.error(f"Max reconnection attempts ({max_reconnect_attempts}) reached. Giving up.")
                self.connection_status.emit(False, "Max reconnect attempts reached")
                break
            
            logger.info(f"Waiting {reconnect_delay} seconds before reconnecting...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
        
        logger.info("Exiting WebSocket connect_and_listen loop.")
    
    def run(self):
        """
        Overridden QThread.run() method. Create a new asyncio event loop
        and run our connect_and_listen coroutine until the thread is stopped.
        """
        self.running = True
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self.connect_and_listen())
        finally:
            # Clean up the loop
            self.loop.close()
            logger.info("WebSocket thread event loop closed.")
    
    def stop(self):
        """
        Signal the thread to stop. We must also stop the asyncio loop
        if it is currently waiting on anything. Calling loop.stop() via
        call_soon_threadsafe ensures that any waiting awaits exit promptly.
        """
        self.running = False
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)
    
    def get_metrics(self, quantity, volatility, fee_tier):
        """Get current trading metrics"""
        return self.processor.calculate_metrics(quantity, volatility, fee_tier)

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.setWindowTitle("Trade Simulator")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Create left panel (input parameters)
        left_panel = QGroupBox("Input Parameters")
        left_layout = QFormLayout()
        
        # Exchange selection
        self.exchange_combo = QComboBox()
        self.exchange_combo.addItem(EXCHANGE)
        left_layout.addRow("Exchange:", self.exchange_combo)
        
        # Asset selection
        self.asset_combo = QComboBox()
        self.asset_combo.addItem(DEFAULT_SYMBOL)
        left_layout.addRow("Spot Asset:", self.asset_combo)
        
        # Order type
        self.order_type_combo = QComboBox()
        self.order_type_combo.addItem("Market")
        left_layout.addRow("Order Type:", self.order_type_combo)
        
        # Quantity
        self.quantity_input = QLineEdit(str(DEFAULT_QUANTITY))
        left_layout.addRow("Quantity (USD):", self.quantity_input)
        
        # Volatility
        self.volatility_input = QLineEdit(str(DEFAULT_VOLATILITY))
        left_layout.addRow("Volatility:", self.volatility_input)
        
        # Fee tier
        self.fee_tier_combo = QComboBox()
        for tier in FEE_TIERS.keys():
            self.fee_tier_combo.addItem(tier)
        left_layout.addRow("Fee Tier:", self.fee_tier_combo)
        
        # Connection status
        self.connection_status = QLabel("Disconnected")
        self.connection_status.setStyleSheet("color: red;")
        left_layout.addRow("Status:", self.connection_status)
        
        # Connect button
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.toggle_connection)
        left_layout.addRow("", self.connect_button)
        
        left_panel.setLayout(left_layout)
        
        # Create right panel (output parameters)
        right_panel = QGroupBox("Output Parameters")
        right_layout = QFormLayout()
        
        # Output labels
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
        
        # Current price
        self.price_label = QLabel("0.0000")
        self.price_label.setFont(QFont("Arial", 14, QFont.Bold))
        right_layout.addRow("Current Price:", self.price_label)
        
        # Orderbook depth display
        self.depth_label = QLabel("Asks: 0 | Bids: 0")
        right_layout.addRow("Orderbook Depth:", self.depth_label)
        
        right_panel.setLayout(right_layout)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 1)
        
        # Set central widget
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Initialize WebSocket thread
        self.ws_thread = WebSocketThread(WEBSOCKET_URL)
        self.ws_thread.data_received.connect(self.update_data)
        self.ws_thread.processing_complete.connect(self.update_latency)
        self.ws_thread.connection_status.connect(self.update_connection_status)
        
        # Timer for updating metrics
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_metrics)
        self.update_timer.start(UPDATE_INTERVAL_MS)  # Update every 100ms
        
        # Connection state
        self.is_connected = False
    
    def toggle_connection(self):
        """Toggle WebSocket connection"""
        if not self.is_connected:
            # Start WebSocket thread
            self.ws_thread.start()
            self.connect_button.setText("Disconnect")
            self.is_connected = True
        else:
            # Stop WebSocket thread
            self.ws_thread.stop()
            self.ws_thread.wait()
            self.connect_button.setText("Connect")
            self.is_connected = False
            self.connection_status.setText("Disconnected")
            self.connection_status.setStyleSheet("color: red;")
    
    def update_connection_status(self, connected, message):
        """Update connection status label"""
        self.connection_status.setText(message)
        if connected:
            self.connection_status.setStyleSheet("color: green;")
        else:
            # color orange for “in the process of reconnecting” or error
            self.connection_status.setStyleSheet("color: orange;")
    
    def update_data(self, data):
        """Update UI with received data"""
        try:
            asks = data.get("asks", [])
            bids = data.get("bids", [])
            
            if asks and bids:
                best_ask = float(asks[0][0])
                best_bid = float(bids[0][0])
                mid_price = (best_ask + best_bid) / 2
                self.price_label.setText(f"{mid_price:.2f}")
                
                # Update depth label
                self.depth_label.setText(f"Asks: {len(asks)} | Bids: {len(bids)}")
        
        except Exception as e:
            logger.error(f"Error updating data: {e}")
    
    def update_latency(self, processing_time):
        """Update latency label"""
        self.latency_label.setText(f"{processing_time*1000:.4f} ms")
    
    def update_metrics(self):
        """Update trading metrics"""
        if not self.is_connected:
            return
        
        try:
            # Get input values
            quantity = float(self.quantity_input.text())
            volatility = float(self.volatility_input.text())
            fee_tier = self.fee_tier_combo.currentText()
            
            # Calculate metrics
            metrics = self.ws_thread.get_metrics(quantity, volatility, fee_tier)
            
            # Update labels
            self.slippage_label.setText(f"{metrics['slippage']:.6f}")
            self.fees_label.setText(f"{metrics['fees']:.6f}")
            self.market_impact_label.setText(f"{metrics['market_impact']:.6f}")
            self.net_cost_label.setText(f"{metrics['net_cost']:.6f}")
            self.maker_taker_label.setText(f"{metrics['maker_taker_ratio']}")
        
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

def main():
    """Main application entry point"""
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()
