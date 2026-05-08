import numpy as np

def simulate_training(
    num_steps=1000,
    batch_size=4096,
    draw_rate=0.9,
    win_rate=0.05,
    loss_rate=0.05,
    draw_penalty=0.1,
    lr=5e-5,
    wd=5e-4,
    momentum=0.9,
    standardize=True
):
    # Logits for 1 legal move and 1 illegal move (simplification)
    # In reality there are ~30 legal and ~4800 illegal.
    logit_legal = 0.0
    logit_illegal = 0.0
    velocity_legal = 0.0
    
    legal_history = []
    illegal_history = []
    
    for step in range(num_steps):
        # Sample outcomes
        outcomes = np.random.choice(
            [-1, 0, 1], 
            size=batch_size, 
            p=[loss_rate, draw_rate, win_rate]
        )
        
        # Apply draw penalty
        z = np.where(outcomes == 0, -draw_penalty, outcomes)
        
        # Advantage (assume v_baseline correctly predicts -draw_penalty for draws)
        # v_baseline for win/loss might be 0 initially.
        v_baseline = np.zeros(batch_size) - draw_penalty * draw_rate
        adv = z - v_baseline
        
        if standardize:
            mean_adv = np.mean(adv)
            std_adv = np.std(adv) + 1e-6
            adv = (adv - mean_adv) / std_adv
            
        # Gradients for the legal logit (REINFORCE)
        # dL/dx = adv * (p - 1) for played move.
        # Assume p is near 0.5 (one legal, one illegal).
        # Actually p is tiny if illegal is huge.
        # For simplicity, use the REINFORCE update: logit += lr * adv
        # (Assuming the played move is always our 'logit_legal')
        
        # Weighted mean gradient for the batch
        grad_legal = -np.mean(adv) # Negative because we want to maximize z*log(p)
        # Wait, the code does totalLoss = weightedPolicy + ...
        # weightedPolicy = mean(adv_standardized * -log(p))
        # grad = mean(adv_standardized * (p - 1))
        
        # Let's just use the update direction:
        # logit_legal += lr * sum(adv_i * (1 - p_i)) / batch_size
        
        p_legal = np.exp(logit_legal) / (np.exp(logit_legal) + np.exp(logit_illegal))
        
        # Update legal logit
        # grad = adv * (p_legal - 1)
        batch_grad = np.mean(adv * (p_legal - 1))
        
        velocity_legal = momentum * velocity_legal + batch_grad
        logit_legal -= lr * velocity_legal
        
        # Weight decay
        logit_legal -= lr * wd * logit_legal
        logit_illegal -= lr * wd * logit_illegal
        
        legal_history.append(logit_legal)
        illegal_history.append(logit_illegal)
        
    return legal_history, illegal_history

# Run simulation
h_legal, h_illegal = simulate_training(num_steps=5000)

print(f"Final logit_legal: {h_legal[-1]:.6f}")
print(f"Final logit_illegal: {h_illegal[-1]:.6f}")
print(f"Final p_legal: {np.exp(h_legal[-1])/(np.exp(h_legal[-1]) + np.exp(h_illegal[-1])):.6f}")

# Without draw penalty
h_legal_no, h_illegal_no = simulate_training(num_steps=5000, draw_penalty=0)
print(f"\nNo draw penalty:")
print(f"Final logit_legal: {h_legal_no[-1]:.6f}")
print(f"Final logit_illegal: {h_illegal_no[-1]:.6f}")
print(f"Final p_legal: {np.exp(h_legal_no[-1])/(np.exp(h_legal_no[-1]) + np.exp(h_illegal_no[-1])):.6f}")

# Without standardization
h_legal_ns, h_illegal_ns = simulate_training(num_steps=5000, standardize=False)
print(f"\nNo standardization:")
print(f"Final logit_legal: {h_legal_ns[-1]:.6f}")
print(f"Final logit_illegal: {h_illegal_ns[-1]:.6f}")
print(f"Final p_legal: {np.exp(h_legal_ns[-1])/(np.exp(h_legal_ns[-1]) + np.exp(h_illegal_ns[-1])):.6f}")
