# --- Training Schedule ---
max_iters = 20_000          # TinyStories needs more "miles" than Shakespeare
lr_decay_iters = 20_000     # Always match max_iters
eval_interval = 500        # Keep this frequent to catch overfitting early
eval_iters = 100           # Lower this to 100 to speed up the eval phase
log_interval = 10
always_save_checkpoint = False

# --- Dataset & Logging ---
dataset = 'tinystories'    # Assumes you ran the tinystories prepare.py
wandb_log = True
wandb_project = 'tinystories-tiny'
out_prefix_dataset = "ts"

# --- Throughput (Optimized for 4090) ---
# Total batch size = batch_size * gradient_accumulation_steps
# We want roughly 128 - 256 total batch size for stability
batch_size = 32            # 4090 can easily handle 32-64 for small models
gradient_accumulation_steps = 4 
block_size = 512           # Stories are longer than Shakespeare snippets; 256 is too short

# --- Model Hyperparameters ---
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1              # Lower dropout for TinyStories compared to Shakespeare

# --- Optimizer ---
learning_rate = 1e-3       # Small models can handle high LRs
min_lr = 1e-4              # learning_rate / 10
warmup_iters = 500         # Longer warmup for a longer run
dtype = 'bfloat16'
