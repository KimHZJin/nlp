from train_t5 import *

args = get_args()
args.finetune = True
args.experiment_name = 'verified'
args.batch_size = 16
args.test_batch_size = 16

# Set checkpoint_dir manually
model_type = 'ft' if args.finetune else 'scr'
args.checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)

# Load test data
_, _, test_loader = load_t5_data(args.batch_size, args.test_batch_size)

# Load best model
model = load_model_from_checkpoint(args, best=True)
model.eval()

# Run test inference
model_sql_path = 'results/t5_ft_experiment_test.sql'
model_record_path = 'records/t5_ft_experiment_test.pkl'
test_inference(args, model, test_loader, model_sql_path, model_record_path)
print("Test predictions saved!")
