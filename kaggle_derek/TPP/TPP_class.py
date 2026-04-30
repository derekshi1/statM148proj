import torch
import torch.nn as nn
import torch.nn.functional as F

#from TPP_train import DEVICE

class NeuralTPP(nn.Module):
    def __init__(self, num_events, embedding_dim, hidden_dim):
        super(NeuralTPP, self).__init__()
        self.embedding = nn.Embedding(num_events, embedding_dim, padding_idx=0)
        
        # Change batch_size_first to batch_first
        self.rnn = nn.LSTM(embedding_dim + 1, hidden_dim, batch_first=True)
        
        self.event_linear = nn.Linear(hidden_dim, num_events)
        self.time_influence = nn.Linear(hidden_dim, 1) 
        self.intensity_bias = nn.Linear(hidden_dim, 1)
    
    def forward(self, event_seq, delta_seq):
        # Embed and concatenate with deltas
        embedded = self.embedding(event_seq)
        rnn_input = torch.cat([embedded, delta_seq.unsqueeze(-1)], dim=-1)
        
        # h: (batch, seq_len, hidden_dim)
        h, _ = self.rnn(rnn_input)
        
        # Predict parameters for intensity function at each step
        v = self.time_influence(h)        # Time decay factor
        alpha = self.intensity_bias(h)    # Base intensity
        event_logits = self.event_linear(h) # Which event is next?
        
        return alpha, v, event_logits

def tpp_nll_loss(alpha, v, event_logits, target_events, target_deltas, mask, device, num_events=37):
    """
    alpha: Base intensity (batch, seq, 1)
    v: Time influence (batch, seq, 1)
    target_deltas: The actual time until the next event
    """
    target_deltas = target_deltas.unsqueeze(-1)

    # The training data uses log1p(delta_t), so treating it as raw time inside
    # exp(alpha + v * t) is both numerically brittle and a mismatch in units.
    # We use a positive, stable intensity and a midpoint-rule integral instead.
    v = 0.1 * torch.tanh(v)
    logit_t = torch.clamp(alpha + (v * target_deltas), min=-20.0, max=20.0)
    lambda_t = F.softplus(logit_t) + 1e-8
    log_lambda = torch.log(lambda_t)

    midpoint_logits = torch.clamp(alpha + (0.5 * v * target_deltas), min=-20.0, max=20.0)
    midpoint_lambda = F.softplus(midpoint_logits) + 1e-8
    integral = midpoint_lambda * target_deltas
    
    # 3. Event Type Loss (Standard Cross Entropy)
    weights = torch.ones(num_events).to(device)    
    # Give ID 28 a much higher weight (e.g., inverse of the 5% frequency)
    weights[28] = 20.0 
    
    # Apply weights to the CrossEntropy
    event_loss = F.cross_entropy(
        event_logits.transpose(1, 2), 
        target_events, 
        weight=weights, 
        reduction='none'
    )
        
    # Combine and Mask
    step_loss = -log_lambda.squeeze(-1) + integral.squeeze(-1) + event_loss
    masked_loss = (step_loss * mask).sum() / mask.sum()

    return masked_loss
