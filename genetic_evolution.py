import subprocess
import time
import os
import random
import threading
import shutil
import json

# Define hyperparameter ranges with integer ranges and more options
hyperparameter_space = {
    'embedding_dim': list(range(50, 501, 50)),  # 50 to 500 with steps of 50
    'latent_dim': list(range(2, 11)),  # 2 to 10
    'lr': [1e-5, 1e-4, 1e-3, 1e-2],
    'batch_size': [64, 128, 256, 512, 1024],
    'max_length': [64, 128, 256, 512],
    'scheduler_type': ['cosine', 'reduce_on_plateau', 'step_lr'],
    'optimizer_type': ['adam', 'sgd', 'rmsprop'],
    'weight_decay': [0, 1e-5, 1e-4, 1e-3],
    'nonlinear_transform': ['relu', 'tanh', 'gelu'],
    'dropout_rate': [0.0, 0.1, 0.2, 0.3],
    'pooling_type': ['mean', 'max', 'attention'],
}

def generate_hyperparameters():
    return {param: random.choice(choices) for param, choices in hyperparameter_space.items()}

def mutate_hyperparameters(parent_hyperparameters):
    hyperparameters = parent_hyperparameters.copy()
    # Randomly mutate one or more of the hyperparameters
    num_mutations = random.randint(1, 3)  # Mutate 1 to 3 hyperparameters
    for _ in range(num_mutations):
        param_to_mutate = random.choice(list(hyperparameters.keys()))
        hyperparameters[param_to_mutate] = random.choice(hyperparameter_space[param_to_mutate])
    return hyperparameters

global_model_counter = 0
def get_new_model_id(parent_model_id=None):
    global global_model_counter
    global_model_counter += 1
    if parent_model_id:
        return f'{parent_model_id}_{global_model_counter}'
    else:
        return f'model_{global_model_counter}'

def launch_model(model_id, hyperparameters, gpu_id):
    command = [
        'python', 'train_pure_vector_embed.py',
        f'--embedding_dim={hyperparameters["embedding_dim"]}',
        f'--latent_dim={hyperparameters["latent_dim"]}',
        f'--lr={hyperparameters["lr"]}',
        f'--batch_size={hyperparameters["batch_size"]}',
        f'--max_length={hyperparameters["max_length"]}',
        f'--scheduler_type={hyperparameters["scheduler_type"]}',
        f'--optimizer_type={hyperparameters["optimizer_type"]}',
        f'--weight_decay={hyperparameters["weight_decay"]}',
        f'--nonlinear_transform={hyperparameters["nonlinear_transform"]}',
        f'--dropout_rate={hyperparameters["dropout_rate"]}',
        f'--pooling_type={hyperparameters["pooling_type"]}',
        f'--model_id={model_id}',
        '--epochs=15',
    ]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    process = subprocess.Popen(command, env=env)
    return process

def read_model_status(model_id):
    status_file = f'./status/{model_id}_status.json'
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            status = json.load(f)
        return status
    else:
        return None

# Initialize population
population_size = 40
models = []
num_gpus = 8  # Number of GPUs available

directories_to_clear = ['results', 'checkpoints', 'status']
for directory in directories_to_clear:
    os.makedirs(directory, exist_ok=True)
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        os.remove(file_path)

# Initialize models
for i in range(population_size):
    hyperparameters = generate_hyperparameters()
    model_id = get_new_model_id()
    gpu_id = i % num_gpus
    process = launch_model(model_id, hyperparameters, gpu_id)
    models.append({
        'id': model_id,
        'process': process,
        'hyperparameters': hyperparameters,
        'epoch': 0,
        'age': 0,  # Number of evaluation rounds survived
        'status': None,
    })

# Evolution loop
epochs_per_evaluation = 3
total_epochs = 15
max_epochs = total_epochs
evaluation_interval = epochs_per_evaluation
max_rounds = total_epochs // epochs_per_evaluation

for eval_round in range(max_rounds):
    print(f"Starting evaluation round {eval_round + 1}/{max_rounds}")
    # Wait for all models to reach the next evaluation point
    while True:
        all_models_ready = True
        for model in models:
            status = read_model_status(model['id'])
            if status is not None and status['epoch'] >= model['epoch'] + epochs_per_evaluation:
                model['status'] = status
            else:
                all_models_ready = False
                break
        if all_models_ready:
            break
        else:
            time.sleep(60)  # Wait a minute before checking again

    # Gather validation accuracies
    model_scores = []
    for model in models:
        val_acc = model['status']['val_acc']
        model_scores.append({'model': model, 'val_acc': val_acc})

    # Adjust selection probabilities based on age
    num_survivors = len(model_scores) // 2
    # Sort models by val_acc
    model_scores.sort(key=lambda x: x['val_acc'], reverse=True)

    # Apply age-based weighting
    survivors = []
    eliminated = []
    young_models = [m for m in model_scores if m['model']['age'] == 0]
    older_models = [m for m in model_scores if m['model']['age'] > 0]

    # Keep all young models
    survivors.extend(young_models)
    remaining_slots = num_survivors - len(young_models)
    if remaining_slots > 0:
        # Fill remaining slots with best older models
        survivors.extend(older_models[:remaining_slots])
        eliminated.extend(older_models[remaining_slots:])
    else:
        # We have more young models than slots; eliminate some young models
        survivors = survivors[:num_survivors]
        eliminated = model_scores[num_survivors:]

    print(f"Eliminating {len(eliminated)} models and creating new offspring.")

    # Kill eliminated models
    for eliminated_model in eliminated:
        process = eliminated_model['model']['process']
        try:
            process.terminate()
            process.wait(timeout=10)
            if process.poll() is None:
                process.kill()
        except Exception as e:
            print(f"Error terminating process {eliminated_model['model']['id']}: {e}")

        # Remove their result files
        val_acc_file = f'./results/{eliminated_model["model"]["id"]}_val_acc.txt'
        if os.path.exists(val_acc_file):
            os.remove(val_acc_file)

        # Remove their checkpoints
        checkpoint_file = f'./checkpoints/{eliminated_model["model"]["id"]}.ckpt'
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        # Remove their status files
        status_file = f'./status/{eliminated_model["model"]["id"]}_status.json'
        if os.path.exists(status_file):
            os.remove(status_file)

    # Generate new models (offspring)
    offspring = []
    for i in range(len(eliminated)):
        parent_model = random.choice(survivors)['model']
        parent_hyperparameters = parent_model['hyperparameters']
        hyperparameters = mutate_hyperparameters(parent_hyperparameters)
        model_id = get_new_model_id(parent_model['id'])
        gpu_id = i % num_gpus
        process = launch_model(model_id, hyperparameters, gpu_id)
        offspring.append({
            'id': model_id,
            'process': process,
            'hyperparameters': hyperparameters,
            'epoch': parent_model['epoch'],
            'age': 0,  # New model
            'status': None,
        })

    # Update models list
    models = [s['model'] for s in survivors] + offspring

    # Update epochs and age for survivors
    for survivor in survivors:
        survivor['model']['epoch'] += epochs_per_evaluation
        survivor['model']['age'] += 1  # Survived another round

# Wait for final models to finish training
print("Final training phase. Waiting for all models to complete.")
for model in models:
    process = model['process']
    process.wait()

# Gather final validation accuracies
final_scores = []
for model in models:
    status = read_model_status(model['id'])
    if status:
        val_acc = status['val_acc']
        final_scores.append({'model': model, 'val_acc': val_acc})
    else:
        final_scores.append({'model': model, 'val_acc': 0.0})

# Sort models by validation accuracy
final_scores.sort(key=lambda x: x['val_acc'], reverse=True)

# Print final results
print("Final model rankings:")
for rank, score in enumerate(final_scores, start=1):
    model_id = score['model']['id']
    val_acc = score['val_acc']
    hyperparameters = score['model']['hyperparameters']
    print(f"Rank {rank}: {model_id}, Val Acc: {val_acc:.4f}, Hyperparameters: {hyperparameters}")

# Optionally, save the best model
best_model = final_scores[0]['model']
best_checkpoint = f'./checkpoints/{best_model["id"]}.ckpt'
best_model_path = f'./best_model.ckpt'
shutil.copyfile(best_checkpoint, best_model_path)
print(f"Best model checkpoint saved to {best_model_path}")
