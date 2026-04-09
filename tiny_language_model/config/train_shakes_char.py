eval_interval = 500
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

wandb_log = True
wandb_project = 'shakespeare'
out_prefix_dataset = "shake"


dataset = 'shakespeare_char'
gradient_accumulation_steps = 8
batch_size = 8
block_size = 256 # context of up to 256 previous characters

dtype = 'bfloat16'
load_meta=True