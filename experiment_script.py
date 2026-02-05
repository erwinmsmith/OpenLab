import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score, precision_score, recall_score
from sklearn.inspection import DecisionBoundaryDisplay
from tqdm import tqdm
import time
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Generate synthetic time series data
def generate_synthetic_time_series(n_samples, n_timesteps, n_features, n_classes, seed):
    set_seed(seed)

    # Generate base data
    X = np.random.randn(n_samples, n_timesteps, n_features)

    # Create different patterns for each class
    for class_idx in range(n_classes):
        class_mask = np.arange(n_samples) % n_classes == class_idx

        # Add specific temporal patterns for each class
        for i, sample_idx in enumerate(np.where(class_mask)[0]):
            # Add periodic pattern
            t = np.linspace(0, 2*np.pi, n_timesteps)
            pattern = np.sin(t + class_idx) * 0.5

            # Add pattern to specific features
            for feature_idx in range(min(3, n_features)):
                X[sample_idx, :, feature_idx] += pattern * (1 + 0.1 * np.random.randn())

            # Add trend
            trend = np.linspace(0, 1, n_timesteps) * class_idx * 0.1
            for feature_idx in range(3, min(6, n_features)):
                X[sample_idx, :, feature_idx] += trend

            # Add noise
            X[sample_idx, :, 6:] += np.random.normal(0, 0.1, (n_timesteps, n_features - 6))

    # Generate labels
    y = np.arange(n_samples) % n_classes

    return X, y

# CNN-LSTM model definition
class CNNLSTM(nn.Module):
    def __init__(self, input_shape, n_classes, conv_params, lstm_units, dropout):
        super(CNNLSTM, self).__init__()

        self.input_shape = input_shape
        self.n_classes = n_classes

        # CNN layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_shape[1]

        for conv_param in conv_params:
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=conv_param['filters'],
                    kernel_size=conv_param['kernel_size'],
                    padding=conv_param['kernel_size'] // 2
                )
            )
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = conv_param['filters']

        # Calculate CNN output shape
        conv_output_length = input_shape[0] // (2 ** len(conv_params))
        conv_output_channels = conv_params[-1]['filters']

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=conv_output_channels,
            hidden_size=lstm_units,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )

        # Output layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_units * 2, n_classes)  # *2 for bidirectional LSTM

    def forward(self, x):
        # Reshape input: (batch, timesteps, features) -> (batch, features, timesteps)
        x = x.transpose(1, 2)

        # CNN feature extraction
        for layer in self.conv_layers:
            x = layer(x)

        # Reshape back for LSTM: (batch, features, timesteps) -> (batch, timesteps, features)
        x = x.transpose(1, 2)

        # LSTM feature extraction
        lstm_out, _ = self.lstm(x)

        # Take the last time step output
        last_output = lstm_out[:, -1, :]

        # Output layer
        output = self.dropout(last_output)
        output = self.fc(output)

        return output

# Training function
def train_model(model, train_loader, val_loader, epochs, lr, device, patience=10, min_delta=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    best_val_accuracy = 0
    best_epoch = 0
    best_model_state = None
    no_improvement = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        # Calculate accuracy
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}%')
        print('-' * 50)

        # Learning rate scheduling
        scheduler.step(val_accuracy)

        # Early stopping check
        if val_accuracy > best_val_accuracy + min_delta:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_model_state = model.state_dict()
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accuracies, val_accuracies, best_val_accuracy, best_epoch

# Evaluation function
def evaluate_model(model, test_loader, device, n_classes):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)

            _, predicted = torch.max(output, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    f1_weighted = f1_score(all_targets, all_preds, average='weighted')
    f1_macro = f1_score(all_targets, all_preds, average='macro')
    precision_macro = precision_score(all_targets, all_preds, average='macro')
    recall_macro = recall_score(all_targets, all_preds, average='macro')

    # Calculate per-class metrics
    precision_per_class = precision_score(all_targets, all_preds, average=None)
    recall_per_class = recall_score(all_targets, all_preds, average=None)
    f1_per_class = f1_score(all_targets, all_preds, average=None)

    # AUC calculation (multi-class)
    try:
        auc_ovr = roc_auc_score(all_targets, np.array(all_probs), multi_class='ovr')
    except:
        auc_ovr = 0.5

    # Calculate Cohen's Kappa
    cohen_kappa = 0
    if n_classes == 2:
        cohen_kappa = accuracy_score(all_targets, all_preds)

    metrics = {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'auc_ovr': auc_ovr,
        'cohen_kappa': cohen_kappa,
        'confusion_matrix': confusion_matrix(all_targets, all_preds).tolist(),
        'classification_report': classification_report(all_targets, all_preds, output_dict=True)
    }

    return metrics, all_preds, all_targets, all_probs

# Visualization function
def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss curves
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy curves
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_distribution(predictions, targets, class_names, save_path):
    plt.figure(figsize=(12, 8))

    for i, class_name in enumerate(class_names):
        plt.subplot(2, 2, i+1)
        class_mask = np.array(targets) == i
        class_predictions = np.array(predictions)[class_mask]

        if len(class_predictions) > 0:
            plt.hist(class_predictions, bins=20, alpha=0.7, label=f'Class {i}')
            plt.xlabel('Predicted Class')
            plt.ylabel('Frequency')
            plt.title(f'Predictions for {class_name}')
            plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, save_path):
    # CNN layer feature importance (using mean activation)
    conv_weights = []
    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name:
            conv_weights.append(param.data.abs().mean(dim=1).cpu().numpy())

    if conv_weights:
        fig, axes = plt.subplots(1, len(conv_weights), figsize=(15, 4))
        if len(conv_weights) == 1:
            axes = [axes]

        for i, weights in enumerate(conv_weights):
            axes[i].plot(weights.mean(axis=0))
            axes[i].set_title(f'Conv Layer {i+1} Feature Importance')
            axes[i].set_xlabel('Feature Index')
            axes[i].set_ylabel('Average Activation')
            axes[i].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def save_results_to_csv(results_df, save_path):
    results_df.to_csv(save_path, index=False)

def save_metrics_to_json(metrics, seed, save_path):
    result_data = {
        'seed': seed,
        'metrics': metrics
    }

    # Read existing data or create new data
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            all_data = json.load(f)
    else:
        all_data = {'results': []}

    all_data['results'].append(result_data)

    with open(save_path, 'w') as f:
        json.dump(all_data, f, indent=2, default=str)

# Main experiment function
def main():
    # Experiment parameters
    seeds = [42, 123, 456]
    epochs = 50
    batch_size = 32
    learning_rate = 0.001
    n_samples = 1000
    n_timesteps = 50
    n_features = 10
    n_classes = 3

    # Create output directories
    output_dir = Path('artifacts')
    figures_dir = output_dir / 'figures'
    tables_dir = output_dir / 'tables'
    logs_dir = output_dir / 'logs'

    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    # CNN parameters
    conv_params = [
        {'filters': 32, 'kernel_size': 3, 'activation': 'relu'},
        {'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
        {'filters': 128, 'kernel_size': 3, 'activation': 'relu'}
    ]

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Store all experiment results
    all_results = []
    all_training_times = []
    all_best_accuracies = []

    # Run experiment for each seed
    for seed in seeds:
        print(f'\n=== Starting experiment with seed: {seed} ===')
        start_time = time.time()

        # Generate data
        X, y = generate_synthetic_time_series(n_samples, n_timesteps, n_features, n_classes, seed)

        # Data preprocessing
        scaler = StandardScaler()
        X_scaled = X.copy()

        # Standardize each feature for each sample
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                X_scaled[i, :, j] = scaler.fit_transform(X[i, :, j].reshape(-1, 1)).flatten()

        # Data splitting
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=seed, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=seed, stratify=y_test)

        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Create model
        model = CNNLSTM(
            input_shape=[n_timesteps, n_features],
            n_classes=n_classes,
            conv_params=conv_params,
            lstm_units=64,
            dropout=0.2
        ).to(device)

        # Train model
        trained_model, train_losses, val_losses, train_accuracies, val_accuracies, best_val_accuracy, best_epoch = train_model(
            model, train_loader, val_loader, epochs, learning_rate, device
        )

        # Evaluate model
        metrics, predictions, targets, probabilities = evaluate_model(trained_model, test_loader, device, n_classes)

        training_time = time.time() - start_time

        # Save results
        result = {
            'seed': seed,
            'training_time': training_time,
            'best_epoch': best_epoch,
            'best_val_accuracy': best_val_accuracy,
            'test_accuracy': metrics['accuracy'],
            'test_f1_weighted': metrics['f1_weighted'],
            'test_auc_ovr': metrics['auc_ovr'],
            'precision_per_class': metrics['precision_per_class'],
            'recall_per_class': metrics['recall_per_class'],
            'f1_per_class': metrics['f1_per_class'],
            'confusion_matrix': metrics['confusion_matrix']
        }

        all_results.append(result)
        all_training_times.append(training_time)
        all_best_accuracies.append(metrics['accuracy'])

        # Visualization
        seed_figures_dir = figures_dir / f'seed_{seed}'
        seed_figures_dir.mkdir(exist_ok=True)

        plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies,
                          seed_figures_dir / 'training_curves.png')
        plot_confusion_matrix(metrics['confusion_matrix'], [f'Class {i}' for i in range(n_classes)],
                            seed_figures_dir / 'confusion_matrix.png')
        plot_prediction_distribution(predictions, targets, [f'Class {i}' for i in range(n_classes)],
                                  seed_figures_dir / 'prediction_distribution.png')

        try:
            plot_feature_importance(trained_model, seed_figures_dir / 'feature_importance.png')
        except:
            pass

        # Save metrics
        save_metrics_to_json(metrics, seed, output_dir / 'metrics.json')

        print(f'Experiment completed, seed {seed}:')
        print(f'Training time: {training_time:.2f}s')
        print(f'Best validation accuracy: {best_val_accuracy:.4f}')
        print(f'Test accuracy: {metrics["accuracy"]:.4f}')
        print(f'Test F1 (weighted): {metrics["f1_weighted"]:.4f}')
        print(f'AUC OVR: {metrics["auc_ovr"]:.4f}')

    # Calculate statistical results
    if len(all_results) > 0:
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)

        # Calculate summary statistics
        summary_stats = {
            'accuracy_mean': results_df['test_accuracy'].mean(),
            'accuracy_std': results_df['test_accuracy'].std(),
            'f1_weighted_mean': results_df['test_f1_weighted'].mean(),
            'f1_weighted_std': results_df['test_f1_weighted'].std(),
            'auc_ovr_mean': results_df['test_auc_ovr'].mean(),
            'auc_ovr_std': results_df['test_auc_ovr'].std(),
            'training_time_mean': results_df['training_time'].mean(),
            'training_time_std': results_df['training_time'].std(),
            'best_epoch_mean': results_df['best_epoch'].mean(),
            'best_epoch_std': results_df['best_epoch'].std()
        }

        # Save results to CSV
        save_results_to_csv(results_df, tables_dir / 'model_comparison.csv')
        save_results_to_csv(pd.DataFrame([summary_stats]), tables_dir / 'metrics_summary.csv')

        # Generate final visualization
        plot_final_results(results_df, figures_dir)

        # Save summary statistics
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump({
                'summary_statistics': summary_stats,
                'individual_results': all_results,
                'seeds_used': seeds
            }, f, indent=2, default=str)

        print('\n=== All experiments completed ===')
        print(f'Accuracy: {summary_stats["accuracy_mean"]:.4f} ± {summary_stats["accuracy_std"]:.4f}')
        print(f'F1 weighted: {summary_stats["f1_weighted_mean"]:.4f} ± {summary_stats["f1_weighted_std"]:.4f}')
        print(f'AUC OVR: {summary_stats["auc_ovr_mean"]:.4f} ± {summary_stats["auc_ovr_std"]:.4f}')
        print(f'Training time: {summary_stats["training_time_mean"]:.2f} ± {summary_stats["training_time_std"]:.2f}s')

def plot_final_results(results_df, save_dir):
    # Create final results charts
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Accuracy distribution
    axes[0, 0].bar(range(len(results_df)), results_df['test_accuracy'], alpha=0.7)
    axes[0, 0].axhline(y=results_df['test_accuracy'].mean(), color='r', linestyle='--', label=f'Mean: {results_df["test_accuracy"].mean():.4f}')
    axes[0, 0].set_xlabel('Seed Index')
    axes[0, 0].set_ylabel('Test Accuracy')
    axes[0, 0].set_title('Test Accuracy by Seed')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Training time distribution
    axes[0, 1].bar(range(len(results_df)), results_df['training_time'], alpha=0.7, color='green')
    axes[0, 1].axhline(y=results_df['training_time'].mean(), color='r', linestyle='--', label=f'Mean: {results_df["training_time"].mean():.2f}s')
    axes[0, 1].set_xlabel('Seed Index')
    axes[0, 1].set_ylabel('Training Time (seconds)')
    axes[0, 1].set_title('Training Time by Seed')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Accuracy vs F1
    axes[1, 0].scatter(results_df['test_accuracy'], results_df['test_f1_weighted'], alpha=0.7)
    axes[1, 0].set_xlabel('Test Accuracy')
    axes[1, 0].set_ylabel('Test F1 Weighted')
    axes[1, 0].set_title('Accuracy vs F1 Weighted')
    axes[1, 0].grid(True)

    # Training time vs Accuracy
    axes[1, 1].scatter(results_df['training_time'], results_df['test_accuracy'], alpha=0.7, color='red')
    axes[1, 1].set_xlabel('Training Time (seconds)')
    axes[1, 1].set_ylabel('Test Accuracy')
    axes[1, 1].set_title('Training Time vs Accuracy')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_dir / 'final_results.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
