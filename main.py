import time
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim


### ----- Functions ----- ###


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=len(df_features.columns[:-8]), out_features=128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output_layer = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.output_layer(x)
        return x

def scale_dataset(X, device):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    return X_tensor

def dnn_training(df, columns, epochs=100):
    df_residual = pd.DataFrame(index=df.index)
    df_predictor = pd.DataFrame(index=df.index)
    
    for col in columns:
    
        # Create dataset with 'y mortality' index
        df_final = df[df.columns[:-8].tolist() + [col]]

        # X and y set for training
        X = df_final.drop(columns=[col])
        X = scale_dataset(X, device)
        y = df_final[col].values.reshape(-1, 1)
        y = torch.tensor(y, dtype=torch.float32, device=device)
        
        # # Convert data to PyTorch tensors
        # X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        # y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        
        # Model training
        model = NeuralNetwork()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # Use trained model to predict values over the X set
        with torch.no_grad():
            y_pred = model(X).numpy()
        df_predictor[col] = y_pred.flatten()
        residues = y.flatten() - y_pred.flatten()
        df_residual[col] = residues
        
        # Calculate errors
        mae, mse = round(mean_absolute_error(y.flatten(), y_pred.flatten()), 2), round(mean_squared_error(y.flatten(), y_pred.flatten()), 2)
        print(f'Mean absolute error: {mae:.2f}', f'\nMean squared error: {mse:.2f}')
                
    return df_predictor, df_residual

### ----- Code ----- ###

import gzip

# Open SIM database
# df_sim = pd.read_csv('compact_mortality_data.csv', header=0, low_memory=False, sep=',')
# df_sim['code_municip'] = df_sim['code_municip'].astype(str)
with gzip.open('sim_data.pkl.gz', 'rb') as f:
    df_sim = pd.read_pickle(f)
print('Total of', len(df_sim), 'municipalities')

# Open PNUD database
with gzip.open('pnud_data.pkl.gz', 'rb') as f:
    df_pnud = pd.read_pickle(f)

# Merge socioeconomic features with mortality indexes
df_features = pd.merge(df_pnud, df_sim, on='code_municip', how='left')
df_features = df_features.set_index('code_municip') # setting 'local_code' to index
print(len(df_features))


### ----- Training step ----- ###

start_time = time.time()
print('Cuda avaiable?', torch.cuda.is_available())
# device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


# # Random Forest training
# columns_rf = df_sim.columns.tolist()[1:]
# rf_predictions, rf_residues = model_training(df_features, columns_rf, num_estimators=10)

# rf_time = time.time()
# print("\n--- %s minutes ---" % ((rf_time - start_time)/60), '\n')

# Deep Neural Network training
columns_dnn = df_sim.columns.tolist()[1:]
df_predictions, df_residues = dnn_training(df_features, columns_dnn, epochs=50)

# dnn_time = time.time()
# print("\n--- %s minutes ---" % ((time.time() - dnn_time)/60))

# Total time
print("\n\nTotal training time: %s minutes" % ((time.time() - start_time)/60))