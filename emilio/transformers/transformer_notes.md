# Next Steps:
1. Create a Hybrid Architecture:
    - 1-Layer CNN at the start (before the Transformer) in order to catch slopes and spikes rather than points- reduce noise
2. Handle class imbalance by penalizing the model for missing the rarer class (in this case, success)
    - code: # If failures are 10x rarer than successes
            weights = torch.tensor([10.0]).to(device)
            criterion = CrossEntropyLoss(weight=weights)
3. Different types of encoding:
    - Sinisuoidal (current implementation): is designed for text data
    - Learnable Positional Encoding:
    - Relative Positional Encoding: Tells the model how far a given step is away from the next step. 
        - This is useful for when journeys are varying lengths. (which is the case for our data)
        - Modifies the Attention Mechanism itself by calculating the distance between relative steps
        
