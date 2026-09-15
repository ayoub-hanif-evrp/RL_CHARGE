import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== GCN MODEL ====================
class GCNModel(nn.Module):
    """Enhanced Graph Convolutional Network for processing station features - GPU OPTIMIZED"""
    
    def __init__(self, hidden_dim=64, output_dim=32):
        super(GCNModel, self).__init__()
        
        # Four GCN layers with increasing then decreasing dimensions
        self.conv1 = nn.Linear(5, hidden_dim)  # Input dimension is 5 (station features)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Linear(hidden_dim, hidden_dim * 2)  # Expand
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.conv3 = nn.Linear(hidden_dim * 2, hidden_dim)  # Compress back
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(0.2)
        
        self.conv4 = nn.Linear(hidden_dim, output_dim)  # Final projection
        self.bn4 = nn.BatchNorm1d(output_dim)
        
        # Residual connection from first to last layer
        self.residual = nn.Linear(5, output_dim)
        
    def forward(self, features, adjacency, training=False):
        """
        Args:
            features: Station features (batch, num_stations, 5)
            adjacency: Adjacency matrix (batch, num_stations, num_stations)
            training: Training mode flag
            
        Returns:
            Station embeddings (batch, num_stations, output_dim)
        """
        # Add self-loops
        batch_size = adjacency.size(0)
        num_stations = adjacency.size(1)
        eye = torch.eye(num_stations, device=adjacency.device).unsqueeze(0).expand(batch_size, -1, -1)
        adjacency = adjacency + eye
        
        # Normalize adjacency matrix (symmetric normalization)
        rowsum = adjacency.sum(dim=2, keepdim=True)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt = torch.where(torch.isinf(d_inv_sqrt), torch.zeros_like(d_inv_sqrt), d_inv_sqrt)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt.squeeze(2))
        adj_normalized = torch.bmm(torch.bmm(d_mat_inv_sqrt, adjacency), d_mat_inv_sqrt)
        
        # Save input for residual connection
        identity = features
        
        # Layer 1
        x = self.conv1(features)
        x = torch.bmm(adj_normalized, x)
        x = x.transpose(1, 2)
        if x.size(2) > 1 or not training:
            x = self.bn1(x)
        x = x.transpose(1, 2)
        x = F.relu(x)
        x = self.dropout1(x) if training else x
        
        # Layer 2 (expansion)
        x = self.conv2(x)
        x = torch.bmm(adj_normalized, x)
        x = x.transpose(1, 2)
        if x.size(2) > 1 or not training:
            x = self.bn2(x)
        x = x.transpose(1, 2)
        x = F.relu(x)
        x = self.dropout2(x) if training else x
        
        # Layer 3 (compression)
        x = self.conv3(x)
        x = torch.bmm(adj_normalized, x)
        x = x.transpose(1, 2)
        if x.size(2) > 1 or not training:
            x = self.bn3(x)
        x = x.transpose(1, 2)
        x = F.relu(x)
        x = self.dropout3(x) if training else x
        
        # Layer 4 (final)
        x = self.conv4(x)
        x = torch.bmm(adj_normalized, x)
        x = x.transpose(1, 2)
        if x.size(2) > 1 or not training:
            x = self.bn4(x)
        x = x.transpose(1, 2)
        
        # Residual connection
        residual_out = self.residual(identity)
        x = x + residual_out
        x = F.relu(x)
        
        return x


# ==================== CUSTOMER CNN ====================
class CustomerCNN(nn.Module):
    """Enhanced CNN for processing customer features with sequential awareness - GPU OPTIMIZED"""
    
    def __init__(self, output_dim=32):
        super(CustomerCNN, self).__init__()
        
        # Input: (batch, num_customers, 4)
        # Treat customers as a sequence with 4 features per customer
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)
        
        # Residual connection
        self.residual_conv = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=1)
        
        # Fully connected layers after pooling
        self.fc1 = nn.Linear(128, 64)  # 64*2 from max+avg pooling
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(64, output_dim)
        self.bn5 = nn.BatchNorm1d(output_dim)
        
    def forward(self, customer_features, training=False):
        """
        Args:
            customer_features: (batch, num_customers, 4)
            training: Training mode flag
            
        Returns:
            Customer embedding (batch, output_dim)
        """
        batch_size = customer_features.size(0)
        num_customers = customer_features.size(1)
        
        # Transpose to (batch, 4, num_customers) for Conv1d
        x = customer_features.transpose(1, 2)  # (batch, 4, num_customers)
        identity = x  # Save for residual connection
        
        # Conv layer 1
        x = self.conv1(x)  # (batch, 32, num_customers)
        if num_customers > 1 or not training:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x) if training else x
        
        # Conv layer 2
        x = self.conv2(x)  # (batch, 64, num_customers)
        if num_customers > 1 or not training:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x) if training else x
        
        # Conv layer 3
        x = self.conv3(x)  # (batch, 64, num_customers)
        if num_customers > 1 or not training:
            x = self.bn3(x)
        
        # Residual connection
        identity = self.residual_conv(identity)  # (batch, 64, num_customers)
        x = x + identity
        x = F.relu(x)
        x = self.dropout3(x) if training else x
        
        # Dual global pooling: max and average
        max_pool = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (batch, 64)
        avg_pool = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (batch, 64)
        x = torch.cat([max_pool, avg_pool], dim=1)  # (batch, 128)
        
        # Fully connected layers
        x = self.fc1(x)
        if batch_size > 1 or not training:
            x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x) if training else x
        
        x = self.fc2(x)
        if batch_size > 1 or not training:
            x = self.bn5(x)
        x = F.relu(x)
        
        return x


# ==================== VEHICLE CNN ====================
class VehicleCNN(nn.Module):
    """
    Enhanced CNN for processing vehicle features - GPU OPTIMIZED
    
    Default input: 7 features per vehicle:
        1. x: Current X position
        2. y: Current Y position  
        3. battery: Current battery level
        4. max_battery: Maximum battery capacity
        5. time: Current time / time elapsed
        6. dist_next: Distance to next customer
        7. remaining_dist: Remaining distance to travel
    """
    
    def __init__(self, input_dim=7, output_dim=32):
        super(VehicleCNN, self).__init__()
        
        # Input: (batch, input_dim) - single vehicle with input_dim features
        # Default is 7 features as specified above
        
        self.input_dim = input_dim
        
        # Fully connected layers to expand dimension
        self.fc1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(64, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(64, output_dim)
        self.bn4 = nn.BatchNorm1d(output_dim)
        
        # Residual connection
        self.residual = nn.Linear(input_dim, output_dim)
        
    def forward(self, vehicle_features, training=False):
        """
        Args:
            vehicle_features: (batch, input_dim) - flexible input dimension
            training: Training mode flag
            
        Returns:
            Vehicle embedding (batch, output_dim)
        """
        batch_size = vehicle_features.size(0)
        
        # Save input for residual
        identity = vehicle_features
        
        # Layer 1
        x = self.fc1(vehicle_features)
        if batch_size > 1 or not training:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x) if training else x
        
        # Layer 2
        x = self.fc2(x)
        if batch_size > 1 or not training:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x) if training else x
        
        # Layer 3
        x = self.fc3(x)
        if batch_size > 1 or not training:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x) if training else x
        
        # Layer 4
        x = self.fc4(x)
        if batch_size > 1 or not training:
            x = self.bn4(x)
        
        # Residual connection
        residual_out = self.residual(identity)
        x = x + residual_out
        x = F.relu(x)
        
        return x


# ==================== STATION SELECTION NETWORK ====================
class StationSelectionNetwork(nn.Module):
    """Network to select charging station - GPU OPTIMIZED"""
    
    def __init__(self, input_dim=128):
        super(StationSelectionNetwork, self).__init__()
        
        # Input: combined features of size input_dim
        # Output: Q-value for each station
        
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(32, 1)  # Output Q-value for each station
        
    def forward(self, combined_features, training=False):
        """
        Args:
            combined_features: (batch, num_stations, input_dim)
            training: Training mode flag
            
        Returns:
            Q-values: (batch, num_stations)
        """
        batch_size = combined_features.size(0)
        num_stations = combined_features.size(1)
        
        # Reshape to process all stations together
        x = combined_features.view(batch_size * num_stations, -1)
        
        # Layer 1
        x = self.fc1(x)
        if num_stations > 1 or not training:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x) if training else x
        
        # Layer 2
        x = self.fc2(x)
        if num_stations > 1 or not training:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x) if training else x
        
        # Layer 3
        x = self.fc3(x)
        if num_stations > 1 or not training:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x) if training else x
        
        # Layer 4 (Q-value output)
        x = self.fc4(x)
        
        # Reshape back to (batch, num_stations)
        q_values = x.view(batch_size, num_stations)
        
        return q_values


# ==================== CHARGE PORTION NETWORK ====================
class ChargePortionNetwork(nn.Module):
    """Network to select charge portion - GPU OPTIMIZED"""
    
    def __init__(self, input_dim=128, num_portions=6):
        super(ChargePortionNetwork, self).__init__()
        
        self.num_portions = num_portions
        
        # Input: combined features of size input_dim
        # Output: Q-value for each portion option
        
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(32, num_portions)  # Output Q-values for all portions
        
    def forward(self, combined_features, training=False):
        """
        Args:
            combined_features: (batch, input_dim)
            training: Training mode flag
            
        Returns:
            Q-values: (batch, num_portions)
        """
        batch_size = combined_features.size(0)
        
        x = combined_features
        
        # Layer 1
        x = self.fc1(x)
        if batch_size > 1 or not training:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x) if training else x
        
        # Layer 2
        x = self.fc2(x)
        if batch_size > 1 or not training:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x) if training else x
        
        # Layer 3
        x = self.fc3(x)
        if batch_size > 1 or not training:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x) if training else x
        
        # Layer 4 (Q-values for portions)
        q_values = self.fc4(x)
        
        return q_values


# ==================== COMPLETE DDQN ARCHITECTURE ====================
class DDQNArchitecture:
    """
    Complete architecture containing all 5 networks - GPU OPTIMIZED:
    - 1 GCN (shared)
    - 1 Customer CNN (shared)
    - 1 Vehicle CNN (shared)
    - 2 Station Networks (Main + Target)
    - 2 Portion Networks (Main + Target)
    """
    
    def __init__(self, device='cpu', vehicle_input_dim=None):
        """
        Initialize all networks and move to specified device
        
        Args:
            device: torch.device or string ('cpu' or 'cuda')
            vehicle_input_dim: Number of vehicle features (auto-detected if None)
        """
        # Handle device
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.vehicle_input_dim = vehicle_input_dim
        
        print(f"Initializing DDQN Architecture on device: {self.device}")
        if vehicle_input_dim is not None:
            print(f"Vehicle input dimension: {vehicle_input_dim}")
        
        # Shared feature extractors - MOVED TO GPU
        self.gcn = GCNModel(hidden_dim=64, output_dim=32).to(self.device)
        self.customer_cnn = CustomerCNN(output_dim=32).to(self.device)
        
        # Vehicle CNN - will be initialized when we know the dimension
        if vehicle_input_dim is not None:
            self.vehicle_cnn = VehicleCNN(input_dim=vehicle_input_dim, output_dim=32).to(self.device)
        else:
            # Default to 7 features: (x, y, battery, max_battery, time, dist_next, remaining_dist)
            self.vehicle_cnn = VehicleCNN(input_dim=7, output_dim=32).to(self.device)
            print("Using default vehicle input dimension: 7 (x, y, battery, max_battery, time, dist_next, remaining_dist)")
        
        # Station selection networks - MOVED TO GPU
        self.main_station_network = StationSelectionNetwork(input_dim=128).to(self.device)
        self.target_station_network = StationSelectionNetwork(input_dim=128).to(self.device)
        
        # Charge portion networks - MOVED TO GPU
        self.main_portion_network = ChargePortionNetwork(input_dim=128, num_portions=6).to(self.device)
        self.target_portion_network = ChargePortionNetwork(input_dim=128, num_portions=6).to(self.device)
        
        print(f"✓ All networks successfully moved to {self.device}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.gcn.parameters()) + \
                      sum(p.numel() for p in self.customer_cnn.parameters()) + \
                      sum(p.numel() for p in self.vehicle_cnn.parameters()) + \
                      sum(p.numel() for p in self.main_station_network.parameters()) + \
                      sum(p.numel() for p in self.main_portion_network.parameters())
        print(f"Total trainable parameters: {total_params:,}")
        
        # Initialize target networks with same weights as main networks
        # self.update_target_networks()
    
    def update_target_networks(self):
        """Copy weights from main networks to target networks"""
        
        # Copy station network weights
        self.target_station_network.load_state_dict(self.main_station_network.state_dict())
        
        # Copy portion network weights
        self.target_portion_network.load_state_dict(self.main_portion_network.state_dict())
    
    def extract_features(self, station_features, adj_matrix, customer_features, 
                        vehicle_features, training=False):
        """
        Extract features using shared networks
        
        Args:
            station_features: (batch, num_stations, 5) - ON GPU
            adj_matrix: (batch, num_stations, num_stations) - ON GPU
            customer_features: (batch, num_customers, 4) - ON GPU
            vehicle_features: (batch, vehicle_dim) - ON GPU (flexible dimension)
            training: Training mode flag
            
        Returns:
            station_embeddings: (batch, num_stations, 32)
            global_station_emb: (batch, 32)
            customer_emb: (batch, 32)
            vehicle_emb: (batch, 32)
        """
        # Auto-detect and re-initialize vehicle CNN if needed
        if self.vehicle_input_dim is None:
            detected_dim = vehicle_features.size(-1)
            if detected_dim != self.vehicle_cnn.input_dim:
                print(f"⚠️  Auto-detected vehicle dimension: {detected_dim} (expected {self.vehicle_cnn.input_dim})")
                print(f"Re-initializing VehicleCNN with correct dimension...")
                self.vehicle_cnn = VehicleCNN(input_dim=detected_dim, output_dim=32).to(self.device)
                self.vehicle_input_dim = detected_dim
                print(f"✓ VehicleCNN re-initialized successfully")
        
        # Process stations with GCN
        station_embeddings = self.gcn(station_features, adj_matrix, training=training)
        
        # Global station embedding (max pooling)
        global_station_emb = station_embeddings.max(dim=1)[0]
        
        # Process customers
        customer_emb = self.customer_cnn(customer_features, training=training)
        
        # Process vehicle
        vehicle_emb = self.vehicle_cnn(vehicle_features, training=training)
        
        return station_embeddings, global_station_emb, customer_emb, vehicle_emb
    
    def get_station_q_values(self, station_features, adj_matrix, customer_features,
                            vehicle_features, use_target=False, training=False):
        """
        Get Q-values for all stations
        
        Args:
            station_features: (batch, num_stations, 5) - ON GPU
            adj_matrix: (batch, num_stations, num_stations) - ON GPU
            customer_features: (batch, num_customers, 4) - ON GPU
            vehicle_features: (batch, 5) - ON GPU
            use_target: Use target network if True
            training: Training mode flag
            
        Returns:
            Q-values for each station (batch, num_stations)
        """
        # Set network modes
        if not training:
            self.gcn.eval()
            self.customer_cnn.eval()
            self.vehicle_cnn.eval()
            if use_target:
                self.target_station_network.eval()
            else:
                self.main_station_network.eval()
        
        batch_size = station_features.size(0)
        num_stations = station_features.size(1)
        
        # Extract features
        station_embs, global_station, customer_emb, vehicle_emb = self.extract_features(
            station_features, adj_matrix, customer_features, vehicle_features, training
        )
        
        # Expand embeddings to match all stations
        global_station_exp = global_station.unsqueeze(1).expand(-1, num_stations, -1)
        customer_emb_exp = customer_emb.unsqueeze(1).expand(-1, num_stations, -1)
        vehicle_emb_exp = vehicle_emb.unsqueeze(1).expand(-1, num_stations, -1)
        
        # Combine: [station_emb, global_station, customer, vehicle]
        combined = torch.cat([
            station_embs,
            global_station_exp,
            customer_emb_exp,
            vehicle_emb_exp
        ], dim=-1)  # (batch, num_stations, 128)
        
        # Get Q-values
        network = self.target_station_network if use_target else self.main_station_network
        q_values = network(combined, training=training)
        
        return q_values
    
    def get_portion_q_values(self, station_features, adj_matrix, customer_features,
                            vehicle_features, selected_station_idx, use_target=False, 
                            training=False):
        """
        Get Q-values for all charge portions given selected station
        
        Args:
            station_features: (batch, num_stations, 5) - ON GPU
            adj_matrix: (batch, num_stations, num_stations) - ON GPU
            customer_features: (batch, num_customers, 4) - ON GPU
            vehicle_features: (batch, 5) - ON GPU
            selected_station_idx: (batch,) indices of selected stations - ON GPU
            use_target: Use target network if True
            training: Training mode flag
            
        Returns:
            Q-values for each portion (batch, num_portions)
        """
        # Set network modes
        if not training:
            self.gcn.eval()
            self.customer_cnn.eval()
            self.vehicle_cnn.eval()
            if use_target:
                self.target_portion_network.eval()
            else:
                self.main_portion_network.eval()
        
        batch_size = station_features.size(0)
        
        # Extract features
        station_embs, global_station, customer_emb, vehicle_emb = self.extract_features(
            station_features, adj_matrix, customer_features, vehicle_features, training
        )
        
        # Get embedding of selected station
        batch_indices = torch.arange(batch_size, dtype=torch.long, device=station_features.device)
        selected_station_emb = station_embs[batch_indices, selected_station_idx]
        
        # Combine: [selected_station_emb, global_station, customer, vehicle]
        combined = torch.cat([
            selected_station_emb,
            global_station,
            customer_emb,
            vehicle_emb
        ], dim=-1)  # (batch, 128)
        
        # Get Q-values
        network = self.target_portion_network if use_target else self.main_portion_network
        q_values = network(combined, training=training)
        
        return q_values
    
    def save_networks(self, filepath_prefix):
        """Save all network weights"""
        torch.save(self.gcn.state_dict(), f'{filepath_prefix}_gcn.pt')
        torch.save(self.customer_cnn.state_dict(), f'{filepath_prefix}_customer_cnn.pt')
        torch.save(self.vehicle_cnn.state_dict(), f'{filepath_prefix}_vehicle_cnn.pt')
        torch.save(self.main_station_network.state_dict(), f'{filepath_prefix}_station_main.pt')
        torch.save(self.main_portion_network.state_dict(), f'{filepath_prefix}_portion_main.pt')
        print(f"All networks saved with prefix: {filepath_prefix}")
    
    def load_networks(self, filepath_prefix):
        """Load all network weights"""
        self.gcn.load_state_dict(torch.load(f'{filepath_prefix}_gcn.pt', map_location=self.device))
        self.customer_cnn.load_state_dict(torch.load(f'{filepath_prefix}_customer_cnn.pt', map_location=self.device))
        self.vehicle_cnn.load_state_dict(torch.load(f'{filepath_prefix}_vehicle_cnn.pt', map_location=self.device))
        self.main_station_network.load_state_dict(torch.load(f'{filepath_prefix}_station_main.pt', map_location=self.device))
        self.main_portion_network.load_state_dict(torch.load(f'{filepath_prefix}_portion_main.pt', map_location=self.device))
        
        # Update target networks
        self.update_target_networks()
        print(f"All networks loaded from prefix: {filepath_prefix}")