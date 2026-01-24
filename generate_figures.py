"""
Script to automatically generate all figures for the LaTeX report.
This script loads the data, trains the model (or loads if exists), and generates all visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

print("="*60)
print("Generating Figures for LaTeX Report")
print("="*60)

# Detect environment and set data paths
if os.path.exists('/kaggle/input'):
    DATA_PATH = '/kaggle/input/heartbeat'
    OUTPUT_PATH = '/kaggle/working'
    print("Running on Kaggle")
else:
    DATA_PATH = 'prac1/data'
    OUTPUT_PATH = '.'
    print("Running locally")

# Load data
print("\n[1/6] Loading data...")
train_df = pd.read_csv(f'{DATA_PATH}/mitbih_train.csv', header=None)
test_df = pd.read_csv(f'{DATA_PATH}/mitbih_test.csv', header=None)
print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Prepare data
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values.astype(int)
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values.astype(int)

class_names = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']

# Preprocess
print("\n[2/6] Preprocessing data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load or train model
model_path = f'{OUTPUT_PATH}/ecg_classifier_model.joblib'
scaler_path = f'{OUTPUT_PATH}/ecg_scaler.joblib'

if os.path.exists(model_path) and os.path.exists(scaler_path):
    print("\n[3/6] Loading trained model...")
    rf_model = joblib.load(model_path)
    print("Model loaded successfully!")
else:
    print("\n[3/6] Training model (this may take a few minutes)...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_model.fit(X_train_scaled, y_train)
    print("Model trained successfully!")
    
    # Save model
    joblib.dump(rf_model, model_path)
    joblib.dump(scaler, scaler_path)
    print("Model saved!")

# Make predictions
print("\n[4/6] Making predictions...")
y_pred_test = rf_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Generate Figure 1: Class Distribution
print("\n[5/6] Generating Figure 1: Class Distribution...")
plt.figure(figsize=(10, 6))
unique, counts = np.unique(y_train, return_counts=True)
bars = plt.bar([class_names[int(i)] for i in unique], counts, 
               color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6'])
plt.xlabel('Class', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.title('Class Distribution in Training Data', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
for i, (u, c) in enumerate(zip(unique, counts)):
    plt.text(i, c + 500, str(c), ha='center', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig('figures/class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/class_distribution.png")

# Generate Figure 2: Sample ECG Signals
print("Generating Figure 2: Sample ECG Signals...")
fig, axes = plt.subplots(5, 1, figsize=(14, 12))
colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']

for i, ax in enumerate(axes):
    idx = np.where(y_train == i)[0][0]
    ax.plot(X_train[idx], color=colors[i], linewidth=1.5)
    ax.set_title(f'Class {i}: {class_names[i]}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Steps', fontsize=11)
    ax.set_ylabel('Amplitude', fontsize=11)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Sample ECG Heartbeats for Each Class', fontsize=14, fontweight='bold', y=0.995)
plt.savefig('figures/ecg_samples.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/ecg_samples.png")

# Generate Figure 3: Confusion Matrix
print("Generating Figure 3: Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred_test)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/confusion_matrix.png")

# Generate Figure 4: Normalized Confusion Matrix
print("Generating Figure 4: Normalized Confusion Matrix...")
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', 
            xticklabels=class_names, yticklabels=class_names,
            vmin=0, vmax=1, cbar_kws={'label': 'Recall'})
plt.title('Normalized Confusion Matrix (Recall per Class)', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('figures/confusion_normalized.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/confusion_normalized.png")

# Generate Figure 5: Feature Importance
print("Generating Figure 5: Feature Importance...")
feature_importance = rf_model.feature_importances_
top_n = 30
indices = np.argsort(feature_importance)[-top_n:][::-1]

plt.figure(figsize=(12, 8))
bars = plt.bar(range(top_n), feature_importance[indices], color='steelblue', edgecolor='navy', linewidth=0.5)
plt.xlabel('Feature Rank', fontsize=12, fontweight='bold')
plt.ylabel('Importance', fontsize=12, fontweight='bold')
plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
plt.xticks(range(top_n), [f'#{i+1}\n({indices[i]})' for i in range(top_n)], rotation=45, ha='right', fontsize=9)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('figures/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/feature_importance.png")

print("\n" + "="*60)
print("All figures generated successfully!")
print("="*60)
print("\nGenerated files:")
print("  - figures/class_distribution.png")
print("  - figures/ecg_samples.png")
print("  - figures/confusion_matrix.png")
print("  - figures/confusion_normalized.png")
print("  - figures/feature_importance.png")
print("\nYou can now compile the LaTeX report with these figures!")

