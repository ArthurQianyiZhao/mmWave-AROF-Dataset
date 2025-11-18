import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from modulation_utils import qam_demod, calculate_ber


class MLPEqualizer:
    """
    MLP-based equalizer for complex signal equalization.
    Uses scikit-learn's MLPRegressor.
    """
    
    def __init__(self, num_taps=15, hidden_layer_size=16, max_iter=1000, 
                 learning_rate_init=0.001, random_state=42):
        """
        Initialize MLP Equalizer.
        
        Args:
            num_taps: Size of the sliding window
            hidden_layer_size: Number of neurons in the hidden layer
            max_iter: Maximum number of iterations
            learning_rate_init: Initial learning rate
            random_state: Random seed for reproducibility
        """
        self.num_taps = num_taps
        self.hidden_layer_size = hidden_layer_size
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.random_state = random_state
        self.tap_delay = num_taps // 2
        
        # Initialize model
        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden_layer_size,),
            activation='relu',
            solver='adam',
            max_iter=max_iter,
            early_stopping=True,
            random_state=random_state,
            verbose=False
        )
        
    def prepare_data(self, received_symbols, total_tx_symbols):
        """
        Prepare sliding window data for training.
        
        Args:
            received_symbols: Nx1 complex array of received symbols
            total_tx_symbols: Nx1 complex array of true transmitted symbols
            
        Returns:
            X_real: Real-valued input features (interleaved I/Q)
            T_real: Real-valued target outputs (I and Q)
        """
        received_symbols = received_symbols.flatten()
        total_tx_symbols = total_tx_symbols.flatten()
        
        num_symbols = len(total_tx_symbols)
        num_examples = num_symbols - 2 * self.tap_delay
        
        X_complex = np.zeros((num_examples, self.num_taps), dtype=np.complex128)
        T_complex = np.zeros((num_examples, 1), dtype=np.complex128)
        
        for k in range(num_examples):
            X_complex[k, :] = received_symbols[k : k + self.num_taps]
            T_complex[k] = total_tx_symbols[k + self.tap_delay]
        
        # Split complex data into interleaved real/imaginary parts
        X_real = np.empty((num_examples, 2 * self.num_taps))
        X_real[:, 0::2] = X_complex.real
        X_real[:, 1::2] = X_complex.imag
        
        T_real = np.hstack([T_complex.real, T_complex.imag])
        
        return X_real, T_real
    
    def train(self, received_symbols, total_tx_symbols, tx_bits, modulation_order, test_size=0.3, verbose=True):
        """
        Train the MLP equalizer model.
        
        Args:
            received_symbols: Nx1 complex array of received symbols
            total_tx_symbols: Nx1 complex array of true transmitted symbols
            tx_bits: Transmitted bits for BER calculation
            modulation_order: QAM modulation order (4, 16, 64, etc.)
            test_size: Fraction of data to use for testing
            verbose: Whether to print training progress
            
        Returns:
            Dictionary containing training results and metrics
        """
        if verbose:
            print("Creating sliding window data for MLP...")
        
        X_real, T_real = self.prepare_data(received_symbols, total_tx_symbols)
        
        # Train/Test Split
        X_train, X_test, T_train, T_test = train_test_split(
            X_real, T_real, test_size=test_size, random_state=self.random_state
        )
        
        if verbose:
            print("Training MLP model...")
        
        # Train model
        self.model.fit(X_train, T_train)
        
        if verbose:
            print("Training complete.")
        
        # Evaluate on test set
        results = self.evaluate(X_test, T_test, tx_bits, modulation_order, verbose=verbose)
        
        return results
    
    def predict(self, X):
        """
        Predict equalized symbols from input features.
        
        Args:
            X: Real-valued input features (interleaved I/Q)
            
        Returns:
            Complex-valued equalized symbols
        """
        predictions = self.model.predict(X)
        return predictions[:, 0] + 1j * predictions[:, 1]
    
    def evaluate(self, X_test, T_test, tx_bits, modulation_order, verbose=True):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test input features
            T_test: Test target outputs
            tx_bits: Transmitted bits for BER calculation
            modulation_order: QAM modulation order
            verbose: Whether to print results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Predict
        T_pred = self.model.predict(X_test)
        
        # Reconstruct complex symbols
        actual_symbols_test = T_test[:, 0] + 1j * T_test[:, 1]
        equalized_symbols_test = T_pred[:, 0] + 1j * T_pred[:, 1]
        
        center_real = X_test[:, 2 * self.tap_delay]
        center_imag = X_test[:, 2 * self.tap_delay + 1]
        received_symbols_test = center_real + 1j * center_imag
        
        # Calculate MSE
        mse_before = np.mean(np.abs(actual_symbols_test - received_symbols_test)**2)
        mse_after = np.mean(np.abs(actual_symbols_test - equalized_symbols_test)**2)
        
        # Calculate R2 Score
        from sklearn.metrics import r2_score
        r2_before = r2_score(T_test, np.column_stack([received_symbols_test.real, received_symbols_test.imag]))
        r2_after = r2_score(T_test, T_pred)
        
        # Calculate BER using proper QAM demodulation
        k = int(np.log2(modulation_order))  # bits per symbol
        # Get corresponding bits for test symbols
        num_test_symbols = len(actual_symbols_test)
        num_test_bits = num_test_symbols * k
        
        # Demodulate symbols to bits
        rx_bits_before = qam_demod(received_symbols_test, modulation_order, unit_power=True)
        rx_bits_after = qam_demod(equalized_symbols_test, modulation_order, unit_power=True)
        tx_bits_actual = qam_demod(actual_symbols_test, modulation_order, unit_power=True)
        
        # Calculate BER
        _, ber_before = calculate_ber(tx_bits_actual, rx_bits_before)
        _, ber_after = calculate_ber(tx_bits_actual, rx_bits_after)
        
        if verbose:
            print("\n--- MLP Results ---")
            print(f"MSE After EQ:  {mse_after:.4f}")
            print(f"R2 After EQ:   {r2_after:.4f}")
            print(f"BER After EQ:  {ber_after:.6e}")
        
        return {
            "model_name": "MLP",
            "eq_symbols": equalized_symbols_test,
            "actual_symbols": actual_symbols_test,
            "received_symbols": received_symbols_test,
            "mse_before": mse_before,
            "mse_after": mse_after,
            "r2_before": r2_before,
            "r2_after": r2_after,
            "ber_before": ber_before,
            "ber_after": ber_after
        }
