import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import Dict, Tuple, List, Any
from neural_networks import DDQNArchitecture

class DDQNAgent:
    """Double DQN Agent for EV charging station selection - GPU OPTIMIZED"""
    
    def __init__(self, 
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 batch_size: int = 64,
                 buffer_size: int = 10000,
                 target_update_freq: int = 10,
                 device: str = None):
        """
        Initialize DDQN Agent with GPU support
        
        Args:
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay per episode
            epsilon_min: Minimum epsilon
            batch_size: Batch size for training
            buffer_size: Replay buffer size
            target_update_freq: Update target networks every N episodes
            device: Device to run on ('cpu', 'cuda', or None for auto-detect)
        """
        # AUTO-DETECT GPU if device not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Print device information
        print(f"\n{'='*60}")
        print(f"DDQN Agent Initialization")
        print(f"{'='*60}")
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            print(f"WARNING: Running on CPU - Training will be slower!")
        print(f"{'='*60}\n")
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Neural networks (contains all 5 networks) - NOW ON GPU
        self.networks = DDQNArchitecture(device=self.device)
        
        # Get all trainable parameters for station network
        station_params = (
            list(self.networks.main_station_network.parameters()) +
            list(self.networks.gcn.parameters()) +
            list(self.networks.customer_cnn.parameters()) +
            list(self.networks.vehicle_cnn.parameters())
        )
        
        # Get all trainable parameters for portion network
        portion_params = (
            list(self.networks.main_portion_network.parameters()) +
            list(self.networks.gcn.parameters()) +
            list(self.networks.customer_cnn.parameters()) +
            list(self.networks.vehicle_cnn.parameters())
        )
        
        # Optimizers (separate for station and portion networks)
        self.station_optimizer = torch.optim.Adam(station_params, lr=learning_rate)
        self.portion_optimizer = torch.optim.Adam(portion_params, lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Episode counter for target network updates
        self.episode_count = 0
        
        # Charge portion options
        # self.charge_portions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.charge_portions = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


        self.target_networks_initialized = False
    
    # ==================== ACTION SELECTION ====================
    
    def select_action(self, state: Dict[str, Any], training: bool = True) -> Tuple[int, int]:
        """
        Select action using epsilon-greedy policy (two-step)
        
        Args:
            state: State dictionary with all features
            training: If True, use epsilon-greedy; if False, pure exploitation
            
        Returns:
            Tuple of (station_index, portion_index)
        """
        num_stations = state['num_stations']
        
        if num_stations == 0:
            return 0, 0  # No feasible stations
        
        # Step 1: Select station
        if training and np.random.random() < self.epsilon:
            # Exploration: random station
            station_idx = np.random.randint(0, num_stations)
        else:
            # Exploitation: best station
            station_idx = self._select_best_station(state)
        
        # Step 2: Select charge portion
        if training and np.random.random() < self.epsilon:
            # Exploration: random portion
            portion_idx = np.random.randint(0, len(self.charge_portions))
        else:
            # Exploitation: best portion for selected station
            portion_idx = self._select_best_portion(state, station_idx)
        
        return station_idx, portion_idx
    
    def _select_best_station(self, state: Dict[str, Any]) -> int:
        """
        Select best station using main network
        
        Args:
            state: State dictionary
            
        Returns:
            Best station index
        """
        # Set networks to eval mode
        self.networks.main_station_network.eval()
        self.networks.gcn.eval()
        self.networks.customer_cnn.eval()
        self.networks.vehicle_cnn.eval()
        
        with torch.no_grad():
            # Convert numpy arrays to tensors and move to GPU - add batch dimension
            station_features = torch.from_numpy(state['station_features']).float().to(self.device).unsqueeze(0)
            adj_matrix = torch.from_numpy(state['adj_matrix']).float().to(self.device).unsqueeze(0)
            customer_features = torch.from_numpy(state['customer_features']).float().to(self.device).unsqueeze(0)
            vehicle_features = torch.from_numpy(state['vehicle_features']).float().to(self.device).unsqueeze(0)
            
            # Get Q-values for all stations
            q_values = self.networks.get_station_q_values(
                station_features,
                adj_matrix,
                customer_features,
                vehicle_features,
                use_target=False,
                training=False
            )
            
            # Select station with highest Q-value
            station_idx = torch.argmax(q_values[0]).item()
        
        return int(station_idx)
    
    def _select_best_portion(self, state: Dict[str, Any], station_idx: int) -> int:
        """
        Select best charge portion for given station
        
        Args:
            state: State dictionary
            station_idx: Selected station index
            
        Returns:
            Best portion index
        """
        # Set networks to eval mode
        self.networks.main_portion_network.eval()
        self.networks.gcn.eval()
        self.networks.customer_cnn.eval()
        self.networks.vehicle_cnn.eval()
        
        with torch.no_grad():
            # Convert numpy arrays to tensors and move to GPU - add batch dimension
            station_features = torch.from_numpy(state['station_features']).float().to(self.device).unsqueeze(0)
            adj_matrix = torch.from_numpy(state['adj_matrix']).float().to(self.device).unsqueeze(0)
            customer_features = torch.from_numpy(state['customer_features']).float().to(self.device).unsqueeze(0)
            vehicle_features = torch.from_numpy(state['vehicle_features']).float().to(self.device).unsqueeze(0)
            selected_station_idx = torch.tensor([station_idx], dtype=torch.long, device=self.device)
            
            # Get Q-values for all portions
            q_values = self.networks.get_portion_q_values(
                station_features,
                adj_matrix,
                customer_features,
                vehicle_features,
                selected_station_idx,
                use_target=False,
                training=False
            )
            
            # Select portion with highest Q-value
            portion_idx = torch.argmax(q_values[0]).item()
        
        return int(portion_idx)
    
    # ==================== EXPERIENCE REPLAY ====================
    
    def store_experience(self, state: Dict[str, Any], station_idx: int, portion_idx: int,
                        reward: float, next_state: Dict[str, Any], done: bool):
        """
        Store experience in replay buffer
        
        Args:
            state: Current state
            station_idx: Selected station index
            portion_idx: Selected portion index
            reward: Reward received
            next_state: Next state
            done: Episode done flag
        """
        # Store raw numpy arrays (NOT converted to tensors yet for memory efficiency)
        self.replay_buffer.append({
            'state': state,
            'station_idx': station_idx,
            'portion_idx': portion_idx,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def sample_batch(self) -> List[Dict]:
        """Sample random batch from replay buffer"""
        return random.sample(self.replay_buffer, self.batch_size)
    
    # ==================== TRAINING ====================
    
    def train(self) -> Tuple[float, float]:
        """
        Train both station and portion networks using DDQN
        
        Returns:
            Tuple of (station_loss, portion_loss)
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        # Sample batch
        batch = self.sample_batch()
        
        # Separate data by components
        station_features_list = []
        adj_matrix_list = []
        customer_features_list = []
        vehicle_features_list = []
        station_indices = []
        portion_indices = []
        rewards = []
        next_station_features_list = []
        next_adj_matrix_list = []
        next_customer_features_list = []
        next_vehicle_features_list = []
        dones = []
        
        for experience in batch:
            station_features_list.append(experience['state']['station_features'])
            adj_matrix_list.append(experience['state']['adj_matrix'])
            customer_features_list.append(experience['state']['customer_features'])
            vehicle_features_list.append(experience['state']['vehicle_features'])
            station_indices.append(experience['station_idx'])
            portion_indices.append(experience['portion_idx'])
            rewards.append(experience['reward'])
            
            # Handle None next_state (terminal states)
            if experience['next_state'] is None or experience['done']:
                # Use current state as dummy next_state (won't be used in training since done=True)
                next_station_features_list.append(experience['state']['station_features'])
                next_adj_matrix_list.append(experience['state']['adj_matrix'])
                next_customer_features_list.append(experience['state']['customer_features'])
                next_vehicle_features_list.append(experience['state']['vehicle_features'])
            else:
                next_station_features_list.append(experience['next_state']['station_features'])
                next_adj_matrix_list.append(experience['next_state']['adj_matrix'])
                next_customer_features_list.append(experience['next_state']['customer_features'])
                next_vehicle_features_list.append(experience['next_state']['vehicle_features'])
            
            dones.append(1.0 if experience['done'] else 0.0)
        
        # Train station network
        station_loss = self._train_station_network(
            station_features_list, adj_matrix_list,
            customer_features_list, vehicle_features_list,
            station_indices, rewards,
            next_station_features_list, next_adj_matrix_list,
            next_customer_features_list, next_vehicle_features_list, dones
        )
        
        # Train portion network
        portion_loss = self._train_portion_network(
            station_features_list, adj_matrix_list,
            customer_features_list, vehicle_features_list,
            station_indices, portion_indices, rewards,
            next_station_features_list, next_adj_matrix_list,
            next_customer_features_list, next_vehicle_features_list, dones
        )
        
        return station_loss, portion_loss
    
    def _train_station_network(self, station_features_list, adj_matrix_list,
                           customer_features_list, vehicle_features_list,
                           station_indices, rewards,
                           next_station_features_list, next_adj_matrix_list,
                           next_customer_features_list, next_vehicle_features_list, dones):
        """Train station selection network using Double DQN - GPU OPTIMIZED"""
        
        self.networks.main_station_network.train()
        self.networks.gcn.train()
        self.networks.customer_cnn.train()
        self.networks.vehicle_cnn.train()
        
        total_loss = 0.0
        
        # Process each experience individually (due to variable sizes)
        for i in range(len(station_indices)):
            # Convert to tensors and move to GPU
            station_feat = torch.from_numpy(station_features_list[i]).float().to(self.device).unsqueeze(0)
            adj_mat = torch.from_numpy(adj_matrix_list[i]).float().to(self.device).unsqueeze(0)
            cust_feat = torch.from_numpy(customer_features_list[i]).float().to(self.device).unsqueeze(0)
            veh_feat = torch.from_numpy(vehicle_features_list[i]).float().to(self.device).unsqueeze(0)
            
            self.station_optimizer.zero_grad()
            
            # Current Q-value for selected station
            q_values = self.networks.get_station_q_values(
                station_feat, adj_mat, cust_feat, veh_feat,
                use_target=False, training=True
            )
            current_q = q_values[0, station_indices[i]]
            
            # Compute target Q-value
            if dones[i]:
                target_q = torch.tensor(rewards[i], device=self.device)
            else:
                with torch.no_grad():
                    # Move next state tensors to GPU
                    next_station_feat = torch.from_numpy(next_station_features_list[i]).float().to(self.device).unsqueeze(0)
                    next_adj_mat = torch.from_numpy(next_adj_matrix_list[i]).float().to(self.device).unsqueeze(0)
                    next_cust_feat = torch.from_numpy(next_customer_features_list[i]).float().to(self.device).unsqueeze(0)
                    next_veh_feat = torch.from_numpy(next_vehicle_features_list[i]).float().to(self.device).unsqueeze(0)
                    
                    # Double DQN: Use main network to select action
                    next_q_main = self.networks.get_station_q_values(
                        next_station_feat, next_adj_mat, next_cust_feat, next_veh_feat,
                        use_target=False, training=False
                    )
                    best_next_station = torch.argmax(next_q_main[0])
                    
                    # Use target network to evaluate action
                    next_q_target = self.networks.get_station_q_values(
                        next_station_feat, next_adj_mat, next_cust_feat, next_veh_feat,
                        use_target=True, training=False
                    )
                    
                    target_q = rewards[i] + self.gamma * next_q_target[0, best_next_station]
            
            loss = F.mse_loss(current_q, target_q)
            loss.backward()
            self.station_optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(station_indices)
    
    def _train_portion_network(self, station_features_list, adj_matrix_list,
                               customer_features_list, vehicle_features_list,
                               station_indices, portion_indices, rewards,
                               next_station_features_list, next_adj_matrix_list,
                               next_customer_features_list, next_vehicle_features_list, dones):
        """Train charge portion network using Double DQN - GPU OPTIMIZED"""
        
        self.networks.main_portion_network.train()
        self.networks.gcn.train()
        self.networks.customer_cnn.train()
        self.networks.vehicle_cnn.train()
        
        total_loss = 0.0
        
        for i in range(len(portion_indices)):
            # Convert to tensors and move to GPU
            station_feat = torch.from_numpy(station_features_list[i]).float().to(self.device).unsqueeze(0)
            adj_mat = torch.from_numpy(adj_matrix_list[i]).float().to(self.device).unsqueeze(0)
            cust_feat = torch.from_numpy(customer_features_list[i]).float().to(self.device).unsqueeze(0)
            veh_feat = torch.from_numpy(vehicle_features_list[i]).float().to(self.device).unsqueeze(0)
            station_idx_tensor = torch.tensor([station_indices[i]], dtype=torch.long, device=self.device)
            
            self.portion_optimizer.zero_grad()
            
            # Current Q-value for selected portion
            q_values = self.networks.get_portion_q_values(
                station_feat, adj_mat, cust_feat, veh_feat,
                station_idx_tensor,
                use_target=False, training=True
            )
            current_q = q_values[0, portion_indices[i]]
            
            # Compute target Q-value
            if dones[i]:
                target_q = torch.tensor(rewards[i], device=self.device)
            else:
                with torch.no_grad():
                    # Move next state tensors to GPU
                    next_station_feat = torch.from_numpy(next_station_features_list[i]).float().to(self.device).unsqueeze(0)
                    next_adj_mat = torch.from_numpy(next_adj_matrix_list[i]).float().to(self.device).unsqueeze(0)
                    next_cust_feat = torch.from_numpy(next_customer_features_list[i]).float().to(self.device).unsqueeze(0)
                    next_veh_feat = torch.from_numpy(next_vehicle_features_list[i]).float().to(self.device).unsqueeze(0)
                    
                    # Double DQN: Use main network to select next station
                    next_q_station_main = self.networks.get_station_q_values(
                        next_station_feat, next_adj_mat, next_cust_feat, next_veh_feat,
                        use_target=False, training=False
                    )
                    best_next_station = torch.argmax(next_q_station_main[0])
                    
                    # Use main network to select next portion
                    next_q_portion_main = self.networks.get_portion_q_values(
                        next_station_feat, next_adj_mat, next_cust_feat, next_veh_feat,
                        best_next_station.unsqueeze(0),
                        use_target=False, training=False
                    )
                    best_next_portion = torch.argmax(next_q_portion_main[0])
                    
                    # Use target network to evaluate
                    next_q_portion_target = self.networks.get_portion_q_values(
                        next_station_feat, next_adj_mat, next_cust_feat, next_veh_feat,
                        best_next_station.unsqueeze(0),
                        use_target=True, training=False
                    )
                    
                    target_q = rewards[i] + self.gamma * next_q_portion_target[0, best_next_portion]
            
            loss = F.mse_loss(current_q, target_q)
            loss.backward()
            self.portion_optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(portion_indices)
    
    # ==================== EPISODE MANAGEMENT ====================
    
    def end_episode(self):
        """Call at end of episode to update counters and target networks"""
        self.episode_count += 1
        
        # Update target networks periodically
        if self.episode_count % self.target_update_freq == 0:
            self.networks.update_target_networks()
            # print(f"  Target networks updated (episode {self.episode_count})")
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    # ==================== SAVE/LOAD ====================
    
    def save(self, filepath_prefix: str):
        """
        Save all networks
        
        Args:
            filepath_prefix: Prefix for saved files
        """
        self.networks.save_networks(filepath_prefix)
    
    def load(self, filepath_prefix: str):
        """
        Load all networks
        
        Args:
            filepath_prefix: Prefix for saved files
        """
        self.networks.load_networks(filepath_prefix)
    
    # ==================== UTILITIES ====================
    
    def get_buffer_size(self) -> int:
        """Get current replay buffer size"""
        return len(self.replay_buffer)
    
    def get_epsilon(self) -> float:
        """Get current epsilon value"""
        return self.epsilon
    
    def set_epsilon(self, epsilon: float):
        """Set epsilon value (for evaluation)"""
        self.epsilon = epsilon
    
    def get_device_info(self) -> str:
        """Get device information for logging"""
        if self.device.type == 'cuda':
            return f"GPU: {torch.cuda.get_device_name(0)}"
        else:
            return "CPU"