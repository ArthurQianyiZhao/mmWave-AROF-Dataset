import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from modulation_utils import qam_demod, calculate_ber


class LSTMEqualizerNet(nn.Module):
    """LSTM network for signal equalization."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=2, dropout=0.2):
        """
        Initialize LSTM network.
        
        Args:
            input_size: Number of input features (2 * num_taps)
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            output_size: Number of output features (2 for I and Q)
            dropout: Dropout rate
        """
        super(LSTMEqualizerNet, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
               For equalization: (batch_size, 1, 2*num_taps)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        out = lstm_out[:, -1, :]
        
        # Fully connected layer
        out = self.fc(out)
        
        return out


class LSTMEqualizer:
    """
    LSTM-based equalizer for complex signal equalization.
    """
    
    def __init__(self, num_taps=9, hidden_size=64, num_layers=2, dropout=0.2, 
                 learning_rate=0.001, batch_size=128, num_epochs=50, random_state=42):
        """
        Initialize LSTM Equalizer.
        
        Args:
            num_taps: Size of the sliding window
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            random_state: Random seed for reproducibility
        """
        self.num_taps = num_taps
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.random_state = random_state
        self.tap_delay = num_taps // 2
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Determine device
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Initialize network
        input_size = 2 * num_taps
        self.model = LSTMEqualizerNet(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=2,
            dropout=dropout
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
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
        Train the LSTM equalizer model.
        
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
            print(f"Using device: {self.device}")
            print("Creating sliding window data for LSTM...")
        
        X_real, T_real = self.prepare_data(received_symbols, total_tx_symbols)
        
        # Train/Test Split
        X_train, X_test, T_train, T_test = train_test_split(
            X_real, T_real, test_size=test_size, random_state=self.random_state
        )
        
        # Convert to PyTorch tensors and reshape for LSTM (add sequence dimension)
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1).to(self.device)  # (N, 1, features)
        T_train_tensor = torch.FloatTensor(T_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(self.device)
        T_test_tensor = torch.FloatTensor(T_test).to(self.device)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, T_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if verbose:
            print("Training LSTM model...")
        
        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch_X, batch_T in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_T)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(train_loader)
                print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.6f}")
        
        if verbose:
            print("Training complete.")
        
        # Evaluate on test set
        results = self.evaluate(X_test_tensor, T_test_tensor, X_test, T_test, tx_bits, modulation_order, verbose=verbose)
        
        return results
    
    def predict(self, X):
        """
        Predict equalized symbols from input features.
        
        Args:
            X: PyTorch tensor of input features with shape (N, 1, features)
            
        Returns:
            Complex-valued equalized symbols
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X).cpu().numpy()
        return predictions[:, 0] + 1j * predictions[:, 1]
    
    def evaluate(self, X_test_tensor, T_test_tensor, X_test, T_test, tx_bits, modulation_order, verbose=True):
        """
        Evaluate the model on test data.
        
        Args:
            X_test_tensor: Test input tensor
            T_test_tensor: Test target tensor
            X_test: Test input numpy array
            T_test: Test target numpy array
            tx_bits: Transmitted bits for BER calculation
            modulation_order: QAM modulation order
            verbose: Whether to print results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test_tensor).cpu().numpy()
        
        # Reconstruct complex symbols
        actual_symbols_test = T_test[:, 0] + 1j * T_test[:, 1]
        equalized_symbols_test = predictions[:, 0] + 1j * predictions[:, 1]
        
        center_real = X_test[:, 2 * self.tap_delay]
        center_imag = X_test[:, 2 * self.tap_delay + 1]
        received_symbols_test = center_real + 1j * center_imag
        
        # Calculate MSE
        mse_before = np.mean(np.abs(actual_symbols_test - received_symbols_test)**2)
        mse_after = np.mean(np.abs(actual_symbols_test - equalized_symbols_test)**2)
        
        # Calculate R2 Score
        from sklearn.metrics import r2_score
        r2_before = r2_score(T_test, np.column_stack([received_symbols_test.real, received_symbols_test.imag]))
        r2_after = r2_score(T_test, predictions)
        
        # Calculate BER using proper QAM demodulation
        k = int(np.log2(modulation_order))  # bits per symbol
        
        # Demodulate symbols to bits
        rx_bits_before = qam_demod(received_symbols_test, modulation_order, unit_power=True)
        rx_bits_after = qam_demod(equalized_symbols_test, modulation_order, unit_power=True)
        tx_bits_actual = qam_demod(actual_symbols_test, modulation_order, unit_power=True)
        
        # Calculate BER
        _, ber_before = calculate_ber(tx_bits_actual, rx_bits_before)
        _, ber_after = calculate_ber(tx_bits_actual, rx_bits_after)
        
        if verbose:
            print("\n--- LSTM Results ---")
            print(f"MSE After EQ:  {mse_after:.4f}")
            print(f"R2 After EQ:   {r2_after:.4f}")
            print(f"BER After EQ:  {ber_after:.6e}")
        
        return {
            "model_name": "LSTM",
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
    
    def save_model(self, filepath):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load_model(self, filepath):
        """Load pre-trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
