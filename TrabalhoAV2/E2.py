import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv

# =====================================================
# ============== Configuration ========================
# =====================================================
N_RODADAS = 100  # As per instructions: R = 100
TRAIN_SPLIT = 0.8 # As per instructions: 80% train, 20% test
CLASSES = ['NO', 'DH', 'SL']
RESULTS_DIR = "rna_results"

# Hyperparameters (can be tuned)
ADALINE_LR = 0.01
ADALINE_EPOCHS = 1000 # Reduced from 1000 for faster Monte Carlo, original was 1000
ADALINE_PR = 1e-6 # Adjusted precision, original was 1e-12

MLP_P_ORIGINAL_FEATURES = 6 # Will be set from data
MLP_L = 2 # Number of hidden layers
MLP_Q = [10, 5] # Neurons in hidden layers
MLP_M_OUTPUT_NEURONS = 3 # Number of classes
MLP_LR = 0.01 # User example had 0.1
MLP_EPOCHS = 1000 # Reduced for speed, original was 1000
MLP_PR = 1e-5 # User example had 1e-5

SHOW_EQM_INTERVAL = None # Set to an int (e.g., 50) to print EQM during training, or None to disable

# =====================================================
# ============== Data Handling Functions ==============
# =====================================================
def load_data(filepath):
    X_raw_list = []
    y_str_list = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                X_raw_list.append([float(val) for val in row[:-1]])
                y_str_list.append(row[-1])
            except ValueError as e:
                print(f"Skipping row due to parsing error: {row} - {e}")
                continue
    return np.array(X_raw_list), np.array(y_str_list)

def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Avoid division by zero if a feature has zero variance
    std[std == 0] = 1 
    return (X - mean) / std

def one_hot_encode_labels(y_str, classes):
    mapping = {label: i for i, label in enumerate(classes)}
    # As per instruction: NO = [1 0 0]; DH = [0 1 0]; SL = [0 0 1]
    # This means if class is 'NO', index 0 is 1. If 'DH', index 1 is 1. If 'SL', index 2 is 1.
    # The order in CLASSES array must match this logic.
    # Let's ensure CLASSES = ['NO', 'DH', 'SL'] for this mapping.
    y_one_hot = np.zeros((len(y_str), len(classes)))
    for i, label in enumerate(y_str):
        if label in mapping:
            y_one_hot[i, mapping[label]] = 1
        else:
            print(f"Warning: Unknown label '{label}' encountered during one-hot encoding.")
    return y_one_hot

def add_bias(X):
    return np.insert(X, 0, -1, axis=1) # Insert -1 as the first column for bias

def split_data(X, Y, train_size, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    N = X.shape[0]
    indices = np.random.permutation(N)
    train_end_idx = int(train_size * N)
    
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]
    
    X_train = X_shuffled[:train_end_idx]
    Y_train = Y_shuffled[:train_end_idx]
    X_test = X_shuffled[train_end_idx:]
    Y_test = Y_shuffled[train_end_idx:]
    return X_train, X_test, Y_train, Y_test

# =====================================================
# ============== Adaline Model (OvA) ==================
# =====================================================
class _AdalineBinary:
    def __init__(self, learning_rate=0.01, n_epochs=1000, pr=1e-12):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.pr = pr
        self.weights = None
        self.cost_history = []

    def _linear_activation(self, X):
        return np.dot(X, self.weights)

    def _step_function(self, linear_output):
        return np.where(linear_output >= 0, 1, 0)

    def _calc_eqm(self, X, y_binary):
        N = X.shape[0]
        if N == 0: return 0
        linear_output = self._linear_activation(X)
        errors = y_binary - linear_output
        eqm = np.sum(errors**2) / (2 * N)
        return eqm

    def fit(self, X, y_binary, show_eqm_interval=None):
        p_plus_1 = X.shape[1]
        N = X.shape[0]
        self.weights = np.random.rand(p_plus_1, 1) - 0.5
        self.cost_history = []
        y_binary_reshaped = y_binary.reshape(-1, 1)

        EQM_prev = self._calc_eqm(X, y_binary_reshaped)
        self.cost_history.append(EQM_prev)

        for epoch in range(self.n_epochs):
            for k in range(N):
                x_k = X[k, :].reshape(1, -1)
                y_k_target = y_binary_reshaped[k, 0]
                
                u_k = self._linear_activation(x_k)[0,0]
                e_k = y_k_target - u_k
                self.weights += self.learning_rate * e_k * x_k.T
            
            EQM_current = self._calc_eqm(X, y_binary_reshaped)
            self.cost_history.append(EQM_current)
            
            if show_eqm_interval and epoch % show_eqm_interval == 0:
                print(f"Adaline Binary - Epoch {epoch}, EQM: {EQM_current:.6f}")

            if np.abs(EQM_current - EQM_prev) < self.pr and epoch > 0:
                # print(f"Adaline Binary converged at epoch {epoch}.")
                break
            EQM_prev = EQM_current

    def predict_scores(self, X):
        return self._linear_activation(X)

    def predict(self, X):
        return self._step_function(self.predict_scores(X))

class AdalineClassifierOvA:
    def __init__(self, n_classes, learning_rate=0.01, n_epochs=1000, pr=1e-12):
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.pr = pr
        self.classifiers = [_AdalineBinary(learning_rate, n_epochs, pr) for _ in range(n_classes)]
        self.learning_curves = []

    def fit(self, X_train, Y_train_one_hot, show_eqm_interval=None):
        self.learning_curves = []
        for i in range(self.n_classes):
            y_binary_train = Y_train_one_hot[:, i]
            # print(f"Training Adaline for class {i}...")
            self.classifiers[i].fit(X_train, y_binary_train, show_eqm_interval)
            self.learning_curves.append(self.classifiers[i].cost_history)
    
    def predict_scores(self, X_test):
        scores = np.zeros((X_test.shape[0], self.n_classes))
        for i in range(self.n_classes):
            scores[:, i] = self.classifiers[i].predict_scores(X_test).flatten()
        return scores

    def predict(self, X_test):
        scores = self.predict_scores(X_test)
        return np.argmax(scores, axis=1) # Returns class indices

# =====================================================
# ================ MLP Model ==========================
# =====================================================
class MLP:
    def __init__(self, p_original_features, L=1, Q=[10], m=3, lr=0.01, n_epochs=1000, pr=1e-12):
        self.p_original_features = p_original_features
        self.L = L 
        self.Q = Q 
        self.m = m 
        self.n_epochs = n_epochs
        self.lr = lr 
        self.pr = pr
        
        self.W = [] 
        self.i_net = [] # Renamed from self.i to avoid conflict with loop variables
        self.y_out = [] # Renamed from self.y
        self.delta = [] # Renamed from self.δ
        self.loss_history = []
    
    def __restart__(self):
        self.W = []
        self.i_net = []
        self.y_out = []
        self.delta = []
        self.loss_history = []
        
    def __init_weights__(self):
        self.__restart__()
        p_plus_1 = self.p_original_features + 1 # +1 for bias
        
        # Input layer to first hidden layer
        self.W.append(np.random.random_sample((self.Q[0], p_plus_1)) - 0.5)
        
        # Between hidden layers
        for idx in range(1, self.L):
            self.W.append(np.random.random_sample((self.Q[idx], self.Q[idx-1] + 1)) - 0.5)
        
        # Last hidden layer to output layer
        self.W.append(np.random.random_sample((self.m, self.Q[-1] + 1)) - 0.5)

        self.i_net = [None] * (self.L + 1)
        self.y_out = [None] * (self.L + 1)
        self.delta = [None] * (self.L + 1)
        
    def __sigmoid__(self, x):
        # Clip to avoid overflow/underflow with exp
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    def __sigmoid_derivative__(self, sigmoid_output):
        return sigmoid_output * (1 - sigmoid_output)
    
    def forward(self, x_sample_col_vec):
        # x_sample_col_vec is (p+1 x 1)
        current_input = x_sample_col_vec
        for j in range(self.L + 1):
            self.i_net[j] = np.dot(self.W[j], current_input)
            self.y_out[j] = self.__sigmoid__(self.i_net[j])
            if j < self.L: # For next layer, add bias to output of current hidden layer
                current_input = np.insert(self.y_out[j], 0, -1).reshape(-1,1)
            # For the last layer (output layer), current_input is not used further in forward pass
                            
    def backward(self, x_sample_col_vec, d_col_vec):
        # x_sample_col_vec is (p+1 x 1), d_col_vec is (m x 1)
        last_layer_idx = self.L 
        
        for j in range(last_layer_idx, -1, -1):
            if j == last_layer_idx: # Output layer
                error = d_col_vec - self.y_out[j]
                self.delta[j] = self.__sigmoid_derivative__(self.y_out[j]) * error
                prev_layer_output_with_bias = np.insert(self.y_out[j-1], 0, -1).reshape(-1,1) if j > 0 else x_sample_col_vec
            elif j == 0: # First hidden layer
                # W_b is weights of layer j+1, excluding bias weights: W[j+1][:, 1:]
                W_b_next_layer = self.W[j+1][:, 1:]
                error_prop = np.dot(W_b_next_layer.T, self.delta[j+1])
                self.delta[j] = self.__sigmoid_derivative__(self.y_out[j]) * error_prop
                prev_layer_output_with_bias = x_sample_col_vec
            else: # Intermediate hidden layers
                W_b_next_layer = self.W[j+1][:, 1:]
                error_prop = np.dot(W_b_next_layer.T, self.delta[j+1])
                self.delta[j] = self.__sigmoid_derivative__(self.y_out[j]) * error_prop
                prev_layer_output_with_bias = np.insert(self.y_out[j-1], 0, -1).reshape(-1,1)
            
            # Update weights
            self.W[j] += self.lr * np.dot(self.delta[j], prev_layer_output_with_bias.T)

    def __calc_eqm__(self, X_train_transposed, Y_train_transposed):
        # X_train_transposed is (p+1 x N), Y_train_transposed is (m x N)
        N = X_train_transposed.shape[1]
        if N == 0: return 0
        EQM_sum = 0
        for k in range(N):
            x_k = X_train_transposed[:, k].reshape(-1, 1)
            d_k = Y_train_transposed[:, k].reshape(-1, 1)
            self.forward(x_k)
            y_k_pred = self.y_out[self.L] # Output of the last layer
            EQM_sum += np.sum((d_k - y_k_pred)**2)
        return EQM_sum / (2 * N)
        
    def fit(self, X_train_transposed, Y_train_transposed, show_eqm_interval=None):
        # X_train_transposed is (p+1 x N), Y_train_transposed is (m x N)
        self.__init_weights__()
        N_train = X_train_transposed.shape[1]
        
        EQM_prev = self.__calc_eqm__(X_train_transposed, Y_train_transposed)
        self.loss_history.append(EQM_prev)

        for epoch in range(self.n_epochs):
            # Shuffle data for each epoch
            indices = np.random.permutation(N_train)
            X_shuffled = X_train_transposed[:, indices]
            Y_shuffled = Y_train_transposed[:, indices]
            
            for k in range(N_train):
                x_sample_col = X_shuffled[:, k].reshape(-1, 1)
                d_col = Y_shuffled[:, k].reshape(-1, 1)
                self.forward(x_sample_col)
                self.backward(x_sample_col, d_col)
            
            EQM_current = self.__calc_eqm__(X_train_transposed, Y_train_transposed)
            self.loss_history.append(EQM_current)
            
            if show_eqm_interval and epoch > 0 and epoch % show_eqm_interval == 0:
                print(f"MLP - Epoch {epoch}, EQM: {EQM_current:.6f}")

            if np.abs(EQM_current - EQM_prev) < self.pr and epoch > 0:
                # print(f"MLP converged at epoch {epoch}.")
                break
            EQM_prev = EQM_current
    
    def predict(self, X_test_transposed):
        # X_test_transposed is (p+1 x N_test)
        N_test = X_test_transposed.shape[1]
        Y_pred_transposed = np.zeros((self.m, N_test))
        for k in range(N_test):
            x_sample_col = X_test_transposed[:, k].reshape(-1, 1)
            self.forward(x_sample_col)
            Y_pred_transposed[:, k] = self.y_out[self.L].flatten()
        return Y_pred_transposed # (m x N_test)

# =====================================================
# ============== Evaluation Functions ================
# =====================================================
def calculate_accuracy(Y_true_one_hot, Y_pred_indices):
    # Y_true_one_hot is (N x C)
    # Y_pred_indices is (N,)
    if Y_true_one_hot.shape[0] == 0: return 0
    Y_true_indices = np.argmax(Y_true_one_hot, axis=1)
    return np.mean(Y_true_indices == Y_pred_indices)

def plot_learning_curve(loss_histories, model_name, run_id, save_dir):
    plt.figure(figsize=(10, 6))
    if isinstance(loss_histories, list) and isinstance(loss_histories[0], list): # For Adaline OvA
        for i, history in enumerate(loss_histories):
            plt.plot(history, label=f'Classifier {i+1} EQM')
    else: # For MLP
        plt.plot(loss_histories, label='EQM')
    plt.title(f'Learning Curve: {model_name} - Run {run_id}')
    plt.xlabel('Epoch')
    plt.ylabel('Error Quadrático Médio (EQM)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{model_name}_run_{run_id}_learning_curve.png'))
    plt.close()

def plot_confusion_matrix(Y_true_one_hot, Y_pred_indices, classes, model_name, run_id, save_dir):
    if Y_true_one_hot.shape[0] == 0: return
    Y_true_indices = np.argmax(Y_true_one_hot, axis=1)
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true_idx, pred_idx in zip(Y_true_indices, Y_pred_indices):
        cm[true_idx, pred_idx] += 1
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix: {model_name} - Run {run_id}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(save_dir, f'{model_name}_run_{run_id}_confusion_matrix.png'))
    plt.close()

# =====================================================
# ================= Main Simulation ===================
# =====================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_raw, y_str = load_data('TrabalhoAV2/dados/coluna_vertebral.csv')
    if X_raw.size == 0:
        print("Failed to load data or data is empty. Exiting.")
        return
        
    MLP_P_ORIGINAL_FEATURES = X_raw.shape[1]

    X_normalized = normalize_features(X_raw)
    X_biased = add_bias(X_normalized) # (N x p_original+1)
    Y_one_hot = one_hot_encode_labels(y_str, CLASSES) # (N x C)
    
    n_features_biased = X_biased.shape[1]
    n_classes = Y_one_hot.shape[1]

    adaline_accuracies = []
    mlp_accuracies = []

    adaline_best_run = {'acc': -1, 'data': None, 'run_idx': -1}
    adaline_worst_run = {'acc': 2, 'data': None, 'run_idx': -1}
    mlp_best_run = {'acc': -1, 'data': None, 'run_idx': -1}
    mlp_worst_run = {'acc': 2, 'data': None, 'run_idx': -1}

    print(f"Starting {N_RODADAS} Monte Carlo simulations...")
    for i_rodada in range(N_RODADAS):
        print(f"Rodada {i_rodada + 1}/{N_RODADAS}")
        X_train, X_test, Y_train_oh, Y_test_oh = split_data(X_biased, Y_one_hot, TRAIN_SPLIT, random_state=i_rodada)

        # --- ADALINE OvA ---
        # print("Training ADALINE OvA...")
        adaline_ova = AdalineClassifierOvA(n_classes=n_classes, 
                                           learning_rate=ADALINE_LR, 
                                           n_epochs=ADALINE_EPOCHS, 
                                           pr=ADALINE_PR)
        adaline_ova.fit(X_train, Y_train_oh, show_eqm_interval=SHOW_EQM_INTERVAL)
        Y_pred_adaline_indices = adaline_ova.predict(X_test)
        acc_adaline = calculate_accuracy(Y_test_oh, Y_pred_adaline_indices)
        adaline_accuracies.append(acc_adaline)

        if acc_adaline > adaline_best_run['acc']:
            adaline_best_run['acc'] = acc_adaline
            adaline_best_run['data'] = {'true': Y_test_oh, 'pred': Y_pred_adaline_indices, 'lc': adaline_ova.learning_curves}
            adaline_best_run['run_idx'] = i_rodada + 1
        if acc_adaline < adaline_worst_run['acc']:
            adaline_worst_run['acc'] = acc_adaline
            adaline_worst_run['data'] = {'true': Y_test_oh, 'pred': Y_pred_adaline_indices, 'lc': adaline_ova.learning_curves}
            adaline_worst_run['run_idx'] = i_rodada + 1

        # --- MLP ---
        # print("Training MLP...")
        mlp = MLP(p_original_features=MLP_P_ORIGINAL_FEATURES,
                  L=MLP_L, Q=MLP_Q, m=n_classes,
                  lr=MLP_LR, n_epochs=MLP_EPOCHS, pr=MLP_PR)
        # MLP expects X: (p+1 x N), Y: (C x N)
        mlp.fit(X_train.T, Y_train_oh.T, show_eqm_interval=SHOW_EQM_INTERVAL)
        Y_pred_mlp_raw_transposed = mlp.predict(X_test.T) # (C x N_test)
        Y_pred_mlp_indices = np.argmax(Y_pred_mlp_raw_transposed.T, axis=1) # N_test indices
        acc_mlp = calculate_accuracy(Y_test_oh, Y_pred_mlp_indices)
        mlp_accuracies.append(acc_mlp)

        if acc_mlp > mlp_best_run['acc']:
            mlp_best_run['acc'] = acc_mlp
            mlp_best_run['data'] = {'true': Y_test_oh, 'pred': Y_pred_mlp_indices, 'lc': mlp.loss_history}
            mlp_best_run['run_idx'] = i_rodada + 1
        if acc_mlp < mlp_worst_run['acc']:
            mlp_worst_run['acc'] = acc_mlp
            mlp_worst_run['data'] = {'true': Y_test_oh, 'pred': Y_pred_mlp_indices, 'lc': mlp.loss_history}
            mlp_worst_run['run_idx'] = i_rodada + 1

    print("Simulations finished. Generating results...")
    summary_lines = []
    summary_lines.append("Resultados da Simulação de Redes Neurais Artificiais")
    summary_lines.append("="*50)
    summary_lines.append(f"Número de Rodadas Monte Carlo: {N_RODADAS}")
    summary_lines.append(f"Divisão Treino/Teste: {TRAIN_SPLIT*100}% / {(1-TRAIN_SPLIT)*100}%")
    summary_lines.append("\n")

    summary_lines.append("Desempenho ADALINE (OvA):")
    summary_lines.append(f"  Acurácia Média: {np.mean(adaline_accuracies):.4f}")
    summary_lines.append(f"  Desvio Padrão da Acurácia: {np.std(adaline_accuracies):.4f}")
    summary_lines.append(f"  Melhor Acurácia: {np.max(adaline_accuracies):.4f} (Rodada {adaline_best_run['run_idx']})")
    summary_lines.append(f"  Pior Acurácia: {np.min(adaline_accuracies):.4f} (Rodada {adaline_worst_run['run_idx']})")
    summary_lines.append("\n")

    summary_lines.append("Desempenho MLP:")
    summary_lines.append(f"  Acurácia Média: {np.mean(mlp_accuracies):.4f}")
    summary_lines.append(f"  Desvio Padrão da Acurácia: {np.std(mlp_accuracies):.4f}")
    summary_lines.append(f"  Melhor Acurácia: {np.max(mlp_accuracies):.4f} (Rodada {mlp_best_run['run_idx']})")
    summary_lines.append(f"  Pior Acurácia: {np.min(mlp_accuracies):.4f} (Rodada {mlp_worst_run['run_idx']})")
    summary_lines.append("\n")

    # Table format as requested
    summary_lines.append("Tabela de Resultados (Acurácia):")
    summary_lines.append("| Modelos | Média     | Desvio-Padrão | Maior Valor | Menor Valor |")
    summary_lines.append("|---------|-----------|---------------|-------------|-------------|")
    summary_lines.append(f"| ADALINE | {np.mean(adaline_accuracies):.4f}    | {np.std(adaline_accuracies):.4f}         | {np.max(adaline_accuracies):.4f}      | {np.min(adaline_accuracies):.4f}      |")
    summary_lines.append(f"| MLP     | {np.mean(mlp_accuracies):.4f}    | {np.std(mlp_accuracies):.4f}         | {np.max(mlp_accuracies):.4f}      | {np.min(mlp_accuracies):.4f}      |")
    summary_lines.append("\n")

    # Plots for best/worst runs
    if adaline_best_run['data']:
        plot_learning_curve(adaline_best_run['data']['lc'], 'ADALINE_OvA', f"best_acc_{adaline_best_run['run_idx']}", RESULTS_DIR)
        plot_confusion_matrix(adaline_best_run['data']['true'], adaline_best_run['data']['pred'], CLASSES, 'ADALINE_OvA', f"best_acc_{adaline_best_run['run_idx']}", RESULTS_DIR)
    if adaline_worst_run['data']:
        plot_learning_curve(adaline_worst_run['data']['lc'], 'ADALINE_OvA', f"worst_acc_{adaline_worst_run['run_idx']}", RESULTS_DIR)
        plot_confusion_matrix(adaline_worst_run['data']['true'], adaline_worst_run['data']['pred'], CLASSES, 'ADALINE_OvA', f"worst_acc_{adaline_worst_run['run_idx']}", RESULTS_DIR)

    if mlp_best_run['data']:
        plot_learning_curve(mlp_best_run['data']['lc'], 'MLP', f"best_acc_{mlp_best_run['run_idx']}", RESULTS_DIR)
        plot_confusion_matrix(mlp_best_run['data']['true'], mlp_best_run['data']['pred'], CLASSES, 'MLP', f"best_acc_{mlp_best_run['run_idx']}", RESULTS_DIR)
    if mlp_worst_run['data']:
        plot_learning_curve(mlp_worst_run['data']['lc'], 'MLP', f"worst_acc_{mlp_worst_run['run_idx']}", RESULTS_DIR)
        plot_confusion_matrix(mlp_worst_run['data']['true'], mlp_worst_run['data']['pred'], CLASSES, 'MLP', f"worst_acc_{mlp_worst_run['run_idx']}", RESULTS_DIR)

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    with open(os.path.join(RESULTS_DIR, 'summary_report.txt'), 'w') as f:
        f.write(summary_text)
    
    print(f"Resultados salvos em '{RESULTS_DIR}'")

if __name__ == '__main__':
    main()

