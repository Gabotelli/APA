import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

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
# 2. LIMPIEZA Y OPTIMIZACIÓN (NUEVO)
# ==========================================
print("--- 2. Optimizando Variables ---")

# A. ELIMINAR COLUMNA FANTASMA
if 'BalanceGross' in df_all.columns:
    print("   > Eliminando 'BalanceGross' (todo ceros)")
    df_all.drop(columns=['BalanceGross'], inplace=True)

# B. MONEDA
def clean_currency(x):
    if isinstance(x, str):
        return float(x.replace('$', '').replace(',', '').strip())
    return x
df_all['DisbursementGross'] = df_all['DisbursementGross'].apply(clean_currency)

# C. FECHAS Y AÑOS (Corrección del viajero en el tiempo)
def clean_approval_fy(x):
    if pd.isna(x): return np.nan
    x = str(x).replace('A', '').strip() 
    try: return int(float(x)) 
    except: return np.nan
df_all['ApprovalFY'] = df_all['ApprovalFY'].apply(clean_approval_fy)

date_cols = ['ApprovalDate', 'DisbursementDate']
for col in date_cols:
    df_all[col] = pd.to_datetime(df_all[col], errors='coerce')
    years = df_all[col].dt.year
    # Corregir años > 2025 (ej: 2068 -> 1968)
    years = years.apply(lambda y: y - 100 if y > 2025 else y)
    df_all[f'{col}_Year'] = years
    df_all[f'{col}_Month'] = df_all[col].dt.month

df_all['Days_To_Disbursement'] = (df_all['DisbursementDate'] - df_all['ApprovalDate']).dt.days

# D. LIMPIEZA QUIRÚRGICA (RevLineCr y LowDoc)
# T -> True (1), R/C/S/A -> Rare/No (0)
def clean_revline(x):
    s = str(x).upper()
    if s in ['Y', 'T', '1']: return 1
    if s in ['N', '0']: return 0
    return np.nan 

def clean_lowdoc(x):
    s = str(x).upper()
    if s in ['Y', '1']: return 1
    if s in ['N', '0', 'A', 'S', 'R', 'C']: return 0
    return np.nan

df_all['RevLineCr'] = df_all['RevLineCr'].apply(clean_revline)
df_all['LowDoc'] = df_all['LowDoc'].apply(clean_lowdoc)

# E. NUEVAS VARIABLES "TESORO"
# 1. Franquicia: Códigos 0 y 1 suelen ser "No Franquicia"
df_all['IsFranchise'] = df_all['FranchiseCode'].apply(lambda x: 0 if x <= 1 else 1)

# 2. Empleos Totales y Creación
df_all['TotalJobs'] = df_all['CreateJob'] + df_all['RetainedJob']
df_all['HasCreatedJobs'] = (df_all['CreateJob'] > 0).astype(int)

# 3. NewExist: Limpiar ceros y nulos (moda = 1)
df_all['NewExist'] = df_all['NewExist'].replace({0.0: 1.0, np.nan: 1.0})

# F. VARIABLES ESTÁNDAR
df_all['StateSame'] = (df_all['State'] == df_all['BankState']).astype(int)
if 'DisbursementDate_Year' in df_all.columns:
    df_all['IsRecession'] = df_all['DisbursementDate_Year'].apply(lambda x: 1 if 2005 <= x <= 2008 else 0)

df_all['NoEmp'] = df_all['NoEmp'].replace(0, 1)
df_all['Amount_Per_Emp'] = df_all['DisbursementGross'] / df_all['NoEmp']

# ==========================================
# 3. ESTRATEGIA DE AGRUPACIÓN (TARGET ENCODING)
# ==========================================
# Eliminar State si tiene varianza 0 (verificado en análisis)
if df_all['State'].nunique() <= 1:
    df_all.drop(columns=['State'], inplace=True, errors='ignore')
    cat_cols_low = ['NewExist', 'UrbanRural', 'IsFranchise', 'HasCreatedJobs']
else:
    cat_cols_low = ['State', 'NewExist', 'UrbanRural', 'IsFranchise', 'HasCreatedJobs']

# Agrupar Bancos y Ciudades
def group_rare_labels(df, col, n_top=20):
    if col not in df.columns: return df, None
    top = df[col].value_counts().head(n_top).index.tolist()
    new_col = f'{col}_Grouped'
    df[new_col] = df[col].apply(lambda x: x if x in top else 'Other_Small')
    return df, new_col

df_all, bank_col = group_rare_labels(df_all, 'Bank', 30) 
df_all, city_col = group_rare_labels(df_all, 'City', 60)

# Label Encoding para baja cardinalidad
le = LabelEncoder()
for col in cat_cols_low:
    df_all[col] = df_all[col].astype(str)
    df_all[col] = le.fit_transform(df_all[col])

# Preparar Target Encoding
cat_cols_high = [c for c in [bank_col, city_col, 'BankState', 'FranchiseCode'] if c]
cols_to_drop = ['id', 'LoanNr_ChkDgt', 'Name', 'ApprovalDate', 'DisbursementDate', 
                'is_train', 'Accept', 'Bank', 'City']

df_train = df_all[df_all['is_train'] == 1].copy()
df_test = df_all[df_all['is_train'] == 0].copy()
y = df_train['Accept'].astype(int)

# Aplicar Target Encoding
for col in cat_cols_high:
    mapping = df_train.groupby(col)['Accept'].mean()
    df_train[f'{col}_TE'] = df_train[col].map(mapping)
    df_test[f'{col}_TE'] = df_test[col].map(mapping)
    
    global_mean = y.mean()
    df_train[f'{col}_TE'] = df_train[f'{col}_TE'].fillna(global_mean)
    df_test[f'{col}_TE'] = df_test[f'{col}_TE'].fillna(global_mean)
    cols_to_drop.append(col)

X = df_train.drop(columns=cols_to_drop)
X_test_final = df_test.drop(columns=cols_to_drop)

# Imputación
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test_final), columns=X_test_final.columns)
X_test_imputed['original_index'] = X_test_imputed.index

# ==========================================
# 4. MODELADO BALANCEADO Y ROBUSTO
# ==========================================
print("\n" + "="*50)
print("   ENTRENANDO MODELO OPTIMIZADO (V4)")
print("="*50)

# Usamos XGBoost como motor principal por su robustez
# Configuración específica para evitar overfitting y manejar desbalanceo
model = xgb.XGBClassifier(
    n_estimators=800,
    learning_rate=0.015,       # Aprendizaje lento y seguro
    max_depth=7,
    subsample=0.7,             # Variedad en datos
    colsample_bytree=0.7,      # Variedad en columnas
    min_child_weight=3,        # Evita hojas muy específicas
    n_jobs=-1,
    random_state=42,
    objective='binary:logistic',
    eval_metric='logloss'
)

# Pesos para balanceo de clases
sample_weights = compute_sample_weight(class_weight='balanced', y=y)

# Validación cruzada rápida para verificar F1 antes de generar
X_tr, X_val, y_tr, y_val = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)
sw_tr = compute_sample_weight('balanced', y_tr)

print("Validando internamente...")
model.fit(X_tr, y_tr, sample_weight=sw_tr)
val_preds = model.predict(X_val)
f1_val = f1_score(y_val, val_preds, average='macro')
print(f"F1-Score (Macro) Estimado: {f1_val:.4f}")

# ==========================================
# 5. SUBMISSION FINAL
# ==========================================
print("\nGenerando archivo final...")
# Re-entrenar con TODO el dataset
model.fit(X_imputed, y, sample_weight=sample_weights)

# Predecir Test
X_test_ready = X_test_imputed.drop(columns=['original_index'])
final_preds = model.predict(X_test_ready)

submission = pd.DataFrame({'id': test['id'], 'Accept': final_preds})
submission.to_csv('submission_v4_optimized.csv', index=False)
print("¡Archivo 'submission_v4_optimized.csv' listo para la victoria! 🚀")