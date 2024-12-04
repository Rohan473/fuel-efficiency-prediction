import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras import Input
import optuna
import matplotlib.pyplot as plt

# Load the data from a local file
file_path = r"C:\projects\ml\fuel_efficiency_prediction\cars.data"

# Define column names based on the dataset structure
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 
                'Acceleration', 'Model Year', 'Origin', 'Car Name']

# Read the CSV file, skipping the first row and using custom delimiter
df = pd.read_csv(file_path, sep=r'\s+', names=column_names, quotechar='"', na_values='?', skipinitialspace=True, skiprows=1)

# Convert numeric columns to float
numeric_columns = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 
                   'Acceleration', 'Model Year', 'Origin']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Update numeric columns after transformations
numeric_columns = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 
                   'Acceleration', 'Power_to_Weight_Ratio', 'Car_Age']

# --- Categorical Encoding for 'Origin' ---
# Convert Origin (1, 2, 3) into categorical variables
origin_encoder = OneHotEncoder(sparse_output=False, drop='first')  # Avoid dummy variable trap
origin_encoded = origin_encoder.fit_transform(df[['Origin']])
origin_encoded_df = pd.DataFrame(origin_encoded, columns=origin_encoder.get_feature_names_out(['Origin']))

# Drop the old 'Origin' column and concatenate encoded features
df = pd.concat([df.drop('Origin', axis=1), origin_encoded_df], axis=1)

# --- Feature Engineering ---

df['Manufacturer'] = df['Car Name'].str.split().str[0].str.lower() 
df.drop('Car Name', axis=1, inplace=True)  # Drop original column

# Add derived features
df['Power_to_Weight_Ratio'] = df['Horsepower'] / df['Weight']
df['Car_Age'] = 2024 - (1900 + df['Model Year'])  # Model year starts from 1900

# Handle missing values first
imputer = SimpleImputer(strategy='mean')
df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# Encode 'Manufacturer' column
manufacturer_encoder = OneHotEncoder(sparse_output=False, drop='first')
manufacturer_encoded = manufacturer_encoder.fit_transform(df[['Manufacturer']])
manufacturer_encoded_df = pd.DataFrame(manufacturer_encoded, columns=manufacturer_encoder.get_feature_names_out(['Manufacturer']))

# Drop the old 'Manufacturer' column and concatenate encoded features
df = pd.concat([df.drop('Manufacturer', axis=1), manufacturer_encoded_df], axis=1)

# Drop any redundant features if necessary
df.drop(['Model Year'], axis=1, inplace=True)  # Example: Dropping the redundant original feature

# --- Outlier Detection ---
# Remove extreme outliers using the Interquartile Range (IQR) method
def remove_outliers(df, column, threshold=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for col in ['MPG', 'Weight', 'Horsepower', 'Acceleration']:
    df = remove_outliers(df, col)

# Finalize Features and Target
target = 'MPG'
X = df.drop(target, axis=1)
y = df[target]

# --- Final Check ---
print("\nUpdated Dataset Summary:")
print(df.info())
print(df.describe())
# Split the data first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values after splitting
imputer = SimpleImputer(strategy='mean')
X_train[numeric_columns] = imputer.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = imputer.transform(X_test[numeric_columns])

# Define the numerical columns that need scaling
numerical_columns = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Power_to_Weight_Ratio', 'Car_Age']

# Separate numerical columns from categorical ones before scaling
X_train_num = X_train[numerical_columns]
X_test_num = X_test[numerical_columns]

# Scale only numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_num)
X_test_scaled = scaler.transform(X_test_num)
# Adjust this line to only drop numerical features, not derived features like Power_to_Weight_Ratio, Car_Age
categorical_columns = [col for col in X.columns if col not in numerical_columns]
X_train_cat = X_train[categorical_columns]
X_test_cat = X_test[categorical_columns]

# Recombine the scaled numerical and categorical data
X_train_final = pd.concat([pd.DataFrame(X_train_scaled, columns=numerical_columns), X_train_cat.reset_index(drop=True)], axis=1)
X_test_final = pd.concat([pd.DataFrame(X_test_scaled, columns=numerical_columns), X_test_cat.reset_index(drop=True)], axis=1)

# Ensure all column names are strings
X_train_final.columns = X_train_final.columns.astype(str)
X_test_final.columns = X_test_final.columns.astype(str)
 

# Linear Regression
lr_model = LinearRegression()# Train linear regression on final features
lr_model.fit(X_train_final, y_train)
y_pred_lr = lr_model.predict(X_test_final)


mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\nLinear Regression Results:")
print(f"Mean Squared Error: {mse_lr:.4f}")
print(f"R-squared Score: {r2_lr:.4f}")

# Neural Network
def create_model(trial):
    # Define hyperparameters to optimize
    n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 4)
    n_units = trial.suggest_int('n_units', 32, 512, log=True)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    # Build the model
    model = Sequential()
    model.add(Input(shape=(X_train_scaled.shape[1],)))  # Fixed the warning
    for _ in range(n_hidden_layers):
        model.add(Dense(n_units, activation=activation))
    model.add(Dense(1))  # Output layer
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

# Define the objective function for Optuna
def objective(trial):
    model = create_model(trial)
    
    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train_scaled, y_train, 
        validation_split=0.2, 
        epochs=50, 
        batch_size=32, 
        verbose=0, 
        callbacks=[early_stopping]
    )
    
    # Evaluate on validation data
    val_loss = min(history.history['val_loss'])
    return val_loss

# Use Optuna for Bayesian Optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20, timeout=600)  # Run for 20 trials or 10 minutes

# Get the best hyperparameters
best_params = study.best_params
print("\nBest Hyperparameters for Neural Network:", best_params)

# Train a final model with the best hyperparameters
best_model = create_model(study.best_trial)
history = best_model.fit(
    X_train_scaled, y_train, 
    validation_split=0.2, 
    epochs=100, 
    batch_size=32, 
    verbose=1, 
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
)

y_pred_nn = best_model.predict(X_test_scaled).flatten()
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

print("\nNeural Network Results:")
print(f"Mean Squared Error: {mse_nn:.4f}")
print(f"R-squared Score: {r2_nn:.4f}")

def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Neural Network Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_training_history(history)
def predict_mpg(input_data):
    """
    Predict MPG for new car data using both Linear Regression and Neural Network models.
    
    Parameters:
    input_data (dict): Dictionary containing car features
        Required keys: 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 
                      'Acceleration', 'Origin', 'Manufacturer'
    
    Returns:
    tuple: (linear_regression_prediction, neural_network_prediction)
    """
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Feature engineering
    input_df['Power_to_Weight_Ratio'] = input_df['Horsepower'] / input_df['Weight']
    input_df['Car_Age'] = 2024 - (1900 + input_df['Model Year'])
    
    # Initialize DataFrame with numeric features
    numeric_features = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 
                       'Acceleration', 'Power_to_Weight_Ratio', 'Car_Age']
    processed_data = input_df[numeric_features].copy()
    
    # Handle Origin encoding
    origin_dummy = pd.DataFrame(0, index=input_df.index, 
                              columns=[col for col in X_train_final.columns if col.startswith('Origin_')])
    if input_df['Origin'].iloc[0] > 1:  # Skip if Origin is 1 (reference category)
        col_name = f'Origin_{input_df["Origin"].iloc[0]}'
        if col_name in origin_dummy.columns:
            origin_dummy[col_name] = 1
    
    # Handle Manufacturer encoding
    manufacturer_dummy = pd.DataFrame(0, index=input_df.index, 
                                    columns=[col for col in X_train_final.columns if col.startswith('Manufacturer_')])
    manufacturer_col = f'Manufacturer_{input_df["Manufacturer"].iloc[0].lower()}'
    if manufacturer_col in manufacturer_dummy.columns:
        manufacturer_dummy[manufacturer_col] = 1
    
    # Combine all features
    processed_data = pd.concat([processed_data, origin_dummy, manufacturer_dummy], axis=1)
    
    # Ensure all columns from training data are present
    for col in X_train_final.columns:
        if col not in processed_data.columns:
            processed_data[col] = 0
            
    # Reorder columns to match training data
    processed_data = processed_data[X_train_final.columns]
    
    # Scale numerical features
    numerical_data = processed_data[numerical_columns]
    numerical_scaled = scaler.transform(numerical_data)
    
    # Create final input for both models
    categorical_columns = [col for col in X_train_final.columns if col not in numerical_columns]
    input_final = pd.concat([
        pd.DataFrame(numerical_scaled, columns=numerical_columns, index=processed_data.index),
        processed_data[categorical_columns]
    ], axis=1)
    
    lr_prediction = lr_model.predict(input_final)[0]
    nn_prediction = best_model.predict(numerical_scaled)[0][0]
    
    return lr_prediction, nn_prediction

if __name__ == "__main__":
    sample_car = {
     'Cylinders': 8,
        'Displacement': 400,
        'Horsepower': 175,
        'Weight': 4464,
        'Acceleration': 11.5,
        'Model Year': 71,
        'Origin': 1,  
        'Manufacturer': 'pontiac catalina brougham '
}
    
    try:
        lr_pred, nn_pred = predict_mpg(sample_car)
        
        print("\nPredictions for sample car:")
        print(f"Linear Regression prediction: {lr_pred:.1f} MPG")
        print(f"Neural Network prediction: {nn_pred:.1f} MPG")
        
        print("\nModel Performance Comparison:")
        print("Linear Regression:")
        print(f"MSE: {mse_lr:.4f}")
        print(f"R²: {r2_lr:.4f}")
        print("\nNeural Network:")
        print(f"MSE: {mse_nn:.4f}")
        print(f"R²: {r2_nn:.4f}")
        
        # Feature importance for Linear Regression
        feature_importance = pd.DataFrame({
            'Feature': X_train_final.columns,
            'Importance': np.abs(lr_model.coef_)
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features (Linear Regression):")
        print(feature_importance.head(10))
        
        # Visualize actual vs predicted values
        plt.figure(figsize=(10, 5))
        
        # Linear Regression
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_pred_lr, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual MPG')
        plt.ylabel('Predicted MPG')
        plt.title('Linear Regression: Actual vs Predicted')
        
        # Neural Network
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_pred_nn, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual MPG')
        plt.ylabel('Predicted MPG')
        plt.title('Neural Network: Actual vs Predicted')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        print("Available manufacturers:", [col.split('_')[1] for col in X_train_final.columns if col.startswith('Manufacturer_')])
        print("Available origins:", [col.split('_')[1] for col in X_train_final.columns if col.startswith('Origin_')])
    