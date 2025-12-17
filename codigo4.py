import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# ==========================================
# 1. CARGA DE DATOS
# ==========================================
print("--- 1. Cargando Datos ---")
train = pd.read_csv('train.csv')
test = pd.read_csv('test_nolabel.csv')

train['is_train'] = 1
test['is_train'] = 0
test['Accept'] = np.nan

df_all = pd.concat([train, test], axis=0).reset_index(drop=True)

# ==========================================
# 2. LIMPIEZA Y FEATURE ENGINEERING (Standard)
# ==========================================
print("--- 2. Limpieza Avanzada ---")

# Moneda
def clean_currency(x):
    if isinstance(x, str): return float(x.replace('$', '').replace(',', '').strip())
    return x
for col in ['DisbursementGross', 'BalanceGross']:
    df_all[col] = df_all[col].apply(clean_currency)

# Fechas y Año
def clean_approval_fy(x):
    if pd.isna(x): return np.nan
    x = str(x).replace('A', '').strip() 
    try: return int(float(x)) 
    except: return np.nan
df_all['ApprovalFY'] = df_all['ApprovalFY'].apply(clean_approval_fy)

for col in ['ApprovalDate', 'DisbursementDate']:
    df_all[col] = pd.to_datetime(df_all[col], errors='coerce')
    years = df_all[col].dt.year
    years = years.apply(lambda y: y - 100 if y > 2025 else y) # Corrección futuro
    df_all[f'{col}_Year'] = years

df_all['Days_To_Disbursement'] = (df_all['DisbursementDate'] - df_all['ApprovalDate']).dt.days

# Categóricas
df_all['RevLineCr'] = df_all['RevLineCr'].apply(lambda x: 1 if str(x) in ['Y','T','1'] else (0 if str(x) in ['N','0'] else np.nan))
df_all['LowDoc'] = df_all['LowDoc'].apply(lambda x: 1 if str(x) in ['Y','1'] else (0 if str(x) in ['N','0'] else np.nan))
df_all['StateSame'] = (df_all['State'] == df_all['BankState']).astype(int)
df_all['NoEmp'] = df_all['NoEmp'].replace(0, 1)
df_all['Amount_Per_Emp'] = df_all['DisbursementGross'] / df_all['NoEmp']

# Agrupación de Raros
def group_rare(df, col, n=20):
    top = df[col].value_counts().head(n).index.tolist()
    df[col] = df[col].apply(lambda x: x if x in top else 'Other')
    return df
df_all = group_rare(df_all, 'Bank', 25)
df_all = group_rare(df_all, 'City', 50)

# Encoding
cat_cols = ['Bank', 'City', 'BankState', 'FranchiseCode', 'State', 'NewExist', 'UrbanRural']
le = LabelEncoder()
for col in cat_cols:
    df_all[col] = df_all[col].astype(str)
    df_all[col] = le.fit_transform(df_all[col])

# Preparar X, y
cols_drop = ['id', 'LoanNr_ChkDgt', 'Name', 'ApprovalDate', 'DisbursementDate', 'is_train', 'Accept']
df_train = df_all[df_all['is_train'] == 1].copy()
df_test = df_all[df_all['is_train'] == 0].copy()
y = df_train['Accept'].astype(int)
X = df_train.drop(columns=cols_drop)
X_test_final = df_test.drop(columns=cols_drop)

# Imputación (Necesaria para PCA)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X_test_imputed = imputer.transform(X_test_final)

# Split Validación
X_train, X_val, y_train, y_val = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 3. PIPELINE: SCALER + PCA + MLP
# ==========================================
print("\n--- 3. Entrenando MLP con PCA ---")

# Definimos el Pipeline
# 1. StandardScaler: OBLIGATORIO para PCA y Redes Neuronales
# 2. PCA: Reduce dimensiones (n_components=0.95 mantiene el 95% de la varianza)
# 3. MLPClassifier: La Red Neuronal
mlp_pca_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95, random_state=42)), 
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(128, 64), # Dos capas ocultas
        activation='relu',            # Función de activación estándar
        solver='adam',                # Optimizador eficiente
        alpha=0.0001,                 # Regularización L2
        batch_size=64,
        learning_rate='adaptive',
        max_iter=500,                 # Épocas máximas
        early_stopping=True,          # Parar si no mejora para evitar overfitting
        random_state=42
    ))
])

# Entrenar
mlp_pca_pipe.fit(X_train, y_train)

# ==========================================
# 4. EVALUACIÓN
# ==========================================
print("\nEvaluando Modelo...")
val_preds = mlp_pca_pipe.predict(X_val)
f1 = f1_score(y_val, val_preds, average='macro')

# Ver cuántas componentes usó PCA
n_components = mlp_pca_pipe.named_steps['pca'].n_components_
print(f"PCA redujo las variables originales a {n_components} componentes principales (95% varianza).")

print("-" * 30)
print(f"MLP + PCA F1-Score (Macro): {f1:.4f}")
print("-" * 30)
print(classification_report(y_val, val_preds))

# ==========================================
# 5. GENERAR SUBMISSION
# ==========================================
print("Generando archivo de envío...")
mlp_pca_pipe.fit(X_imputed, y) # Reentrenar con todo
test_preds = mlp_pca_pipe.predict(X_test_imputed)

submission = pd.DataFrame({'id': test['id'], 'Accept': test_preds})
submission.to_csv('submission_mlp_pca.csv', index=False)
print("¡Archivo 'submission_mlp_pca.csv' generado!")