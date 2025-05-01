import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class TimeSeriesTransformer(nn.Module):
    """Transformer-based time series forecasting model with enhanced architecture"""
    def __init__(self, feature_dim, seq_len, num_heads=4, hidden_dim=128, num_layers=2, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Positional encoding to help model understand sequence positions
        self.positional_encoding = self._create_positional_encoding(seq_len, hidden_dim)
        
        # Input projection layer: map feature_dim to hidden_dim
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        
        # Layer normalization before transformer for training stability
        self.pre_norm = nn.LayerNorm(hidden_dim)
        
        # Transformer encoder configuration
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,  # Standard practice is 4x hidden_dim
            dropout=dropout,
            batch_first=True,
            activation="gelu"  # GELU often performs better than ReLU for transformers
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)  # Final layer normalization
        )
        
        # Attention mechanism for sequence aggregation
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Output network with residual connection
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # Reduce dropout in deeper layers
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Initialize parameters
        self._init_weights()
    
    def _create_positional_encoding(self, seq_len, d_model):
        """Create sinusoidal positional encodings"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0:d_model//2]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
            
        # Register as buffer (not part of gradient updates)
        return nn.Parameter(pe, requires_grad=False)
    
    def _init_weights(self):
        """Initialize model weights for better convergence"""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'projection' in name:
                    # Projection layers initialized with smaller values
                    nn.init.normal_(p, mean=0.0, std=0.02)
                else:
                    # Other layers with Xavier initialization
                    nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape [batch_size, seq_len, feature_dim].
            return_attention (bool): Whether to return attention weights.

        Returns:
            Predicted values tensor of shape [batch_size, 1].
            (Optional) Attention weights.
        """
        batch_size = x.size(0)
        
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        pos_enc = self.positional_encoding.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        x = x + pos_enc
        
        # Apply pre-normalization
        x = self.pre_norm(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Apply attention mechanism for sequence aggregation
        attn_weights = torch.softmax(self.attention(x), dim=1)
        context = torch.sum(x * attn_weights, dim=1)
        
        # Generate prediction through output network
        output = self.output_net(context)
        
        if return_attention:
            return output, attn_weights
        return output
    
    def train_epoch(self, X_train, y_train, optimizer, batch_size=32):
        """Train model for one epoch with improved stability
        
        Args:
            X_train: Training inputs
            y_train: Training targets
            optimizer: PyTorch optimizer
            batch_size: Batch size for training
            
        Returns:
            avg_loss: Average loss for the epoch
            accuracy: Direction prediction accuracy
        """
        self.train()  # Set to training mode
        total_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        # Ensure input data is tensor
        X_train = self._ensure_tensor(X_train)
        y_train = self._ensure_tensor(y_train)
        
        # Create data loader for batching
        indices = torch.randperm(len(X_train))
        num_batches = (len(X_train) + batch_size - 1) // batch_size
        
        criterion = nn.MSELoss()
        
        for i in range(num_batches):
            # Get batch data
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_train))
            batch_indices = indices[start_idx:end_idx]
            
            batch_X = X_train[batch_indices].to(self._get_device())
            batch_y = y_train[batch_indices].view(-1, 1).to(self._get_device())
            
            # Forward pass
            outputs = self(batch_X)
            
            # Calculate loss
            loss = criterion(outputs, batch_y)
            
            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate statistics
            batch_size_actual = batch_y.size(0)
            total_loss += loss.item() * batch_size_actual
            correct_preds += torch.sum((torch.sign(outputs) == torch.sign(batch_y)).float()).item()
            total_samples += batch_size_actual
        
        # Calculate average metrics
        avg_loss = total_loss / total_samples
        accuracy = correct_preds / total_samples
        
        return avg_loss, accuracy
    
    def validate(self, X_val, y_val, batch_size=32):
        """Validate model on validation set
        
        Args:
            X_val: Validation inputs
            y_val: Validation targets
            batch_size: Batch size for validation
            
        Returns:
            avg_loss: Average validation loss
            accuracy: Direction prediction accuracy
        """
        self.eval()  # Set to evaluation mode
        total_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        # Ensure input data is tensor
        X_val = self._ensure_tensor(X_val)
        y_val = self._ensure_tensor(y_val)
        
        # Create data loader for batching
        num_batches = (len(X_val) + batch_size - 1) // batch_size
        
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for i in range(num_batches):
                # Get batch data
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_val))
                
                batch_X = X_val[start_idx:end_idx].to(self._get_device())
                batch_y = y_val[start_idx:end_idx].view(-1, 1).to(self._get_device())
                
                # Forward pass
                outputs = self(batch_X)
                
                # Calculate loss
                loss = criterion(outputs, batch_y)
                
                # Accumulate statistics
                batch_size_actual = batch_y.size(0)
                total_loss += loss.item() * batch_size_actual
                correct_preds += torch.sum((torch.sign(outputs) == torch.sign(batch_y)).float()).item()
                total_samples += batch_size_actual
        
        # Calculate average metrics
        avg_loss = total_loss / total_samples
        accuracy = correct_preds / total_samples
        
        return avg_loss, accuracy
    
    def predict(self, X_test, batch_size=64):
        """Generate predictions for test data
        
        Args:
            X_test: Test inputs
            batch_size: Batch size for prediction
            
        Returns:
            predictions: NumPy array of predictions
        """
        self.eval()  # Set to evaluation mode
        
        # Ensure input data is tensor
        X_test = self._ensure_tensor(X_test)
        
        # Create batches for memory efficiency
        num_samples = len(X_test)
        num_batches = (num_samples + batch_size - 1) // batch_size
        predictions = []
        
        with torch.no_grad():
            for i in range(num_batches):
                # Get batch data
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                
                batch_X = X_test[start_idx:end_idx].to(self._get_device())
                
                # Generate predictions
                batch_preds = self(batch_X)
                
                # Collect batch predictions
                predictions.append(batch_preds.cpu().numpy())
        
        # Concatenate all batch predictions
        all_predictions = np.vstack(predictions).flatten()
        return all_predictions
    
    def _ensure_tensor(self, data):
        """Ensure data is a PyTorch tensor"""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        elif isinstance(data, torch.Tensor):
            return data.float()  # Ensure float type
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    def _get_device(self):
        """Get the device the model is on"""
        return next(self.parameters()).device