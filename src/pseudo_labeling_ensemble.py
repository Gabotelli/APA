import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
# Modelos
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_sample_weight

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'results' / 'runs'
RUNS.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. CARGA DE DATOS
# ==========================================
print("--- 1. Cargando Datos ---")
train = pd.read_csv(ROOT / 'data' / 'train.csv')
test = pd.read_csv(ROOT / 'data' / 'test_nolabel.csv')

train['is_train'] = 1
test['is_train'] = 0
test['Accept'] = np.nan

df_all = pd.concat([train, test], axis=0).reset_index(drop=True)

# ==========================================
# 2. FEATURE ENGINEERING (BASE SÓLIDA V19)
# ==========================================
print("--- 2. Feature Engineering ---")

if 'BalanceGross' in df_all.columns: df_all.drop(columns=['BalanceGross'], inplace=True)

def clean_currency(x):
    if isinstance(x, str): return float(x.replace('$', '').replace(',', '').strip())
    return x
df_all['DisbursementGross'] = df_all['DisbursementGross'].apply(clean_currency)

def clean_approval_fy(x):
    if pd.isna(x): return np.nan
    x = str(x).replace('A', '').strip() 
    try: return int(float(x)) 
    except: return np.nan
df_all['ApprovalFY'] = df_all['ApprovalFY'].apply(clean_approval_fy)

for col in ['ApprovalDate', 'DisbursementDate']:
    df_all[col] = pd.to_datetime(df_all[col], errors='coerce')
    years = df_all[col].dt.year
    years = years.apply(lambda y: y - 100 if y > 2025 else y)
    df_all[f'{col}_Year'] = years

df_all['Days_To_Disbursement'] = (df_all['DisbursementDate'] - df_all['ApprovalDate']).dt.days

# NLP Selectivo
df_all['Name'] = df_all['Name'].fillna('').astype(str).str.upper()
df_all['Legal_Entity'] = df_all['Name'].apply(lambda x: 1 if any(w in x for w in ['LLC', 'INC', 'CORP', 'LTD']) else 0)
df_all['Sector_Risk'] = df_all['Name'].apply(lambda x: 1 if any(w in x for w in ['REALTY', 'ESTATE', 'CONST', 'BUILD', 'DEV']) else 0)

# Categóricas
def clean_binary(x):
    return 1 if str(x).upper() in ['Y','T','1'] else 0
df_all['RevLineCr'] = df_all['RevLineCr'].apply(clean_binary)
df_all['LowDoc'] = df_all['LowDoc'].apply(clean_binary)

df_all['StateSame'] = (df_all['State'] == df_all['BankState']).astype(int)
df_all['IsFranchise'] = df_all['FranchiseCode'].apply(lambda x: 0 if x <= 1 else 1)
df_all['NewExist'] = df_all['NewExist'].replace({0.0: 1.0, np.nan: 1.0})

# Ratios
df_all['NoEmp'] = df_all['NoEmp'].replace(0, 1)
df_all['Loan_Per_Emp'] = df_all['DisbursementGross'] / df_all['NoEmp']
df_all['TotalJobs'] = df_all['CreateJob'] + df_all['RetainedJob']
df_all['RevLine_Urban'] = df_all['RevLineCr'] * df_all['UrbanRural']

# ==========================================
# 3. ENCODING
# ==========================================
if df_all['State'].nunique() <= 1: df_all.drop(columns=['State'], inplace=True, errors='ignore')

def group_rare(df, col, n=40):
    if col not in df.columns: return df, None
    top = df[col].value_counts().head(n).index.tolist()
    df[f'{col}_Grouped'] = df[col].apply(lambda x: x if x in top else 'Other')
    return df, f'{col}_Grouped'

df_all, bank_col = group_rare(df_all, 'Bank', 50)
df_all, city_col = group_rare(df_all, 'City', 80)

le = LabelEncoder()
for col in ['NewExist', 'UrbanRural', 'BankState']:
    df_all[col] = df_all[col].astype(str)
    df_all[col] = le.fit_transform(df_all[col])

# Target Encoding
cols_drop = ['id', 'LoanNr_ChkDgt', 'Name', 'ApprovalDate', 'DisbursementDate', 
             'is_train', 'Accept', 'Bank', 'City']
cat_cols_te = [c for c in [bank_col, city_col, 'FranchiseCode'] if c]

df_train = df_all[df_all['is_train'] == 1].copy()
df_test = df_all[df_all['is_train'] == 0].copy()
y = df_train['Accept'].astype(int)

for col in cat_cols_te:
    global_mean = y.mean()
    agg = df_train.groupby(col)['Accept'].agg(['count', 'mean'])
    smooth = (agg['count'] * agg['mean'] + 20 * global_mean) / (agg['count'] + 20)
    df_train[f'{col}_TE'] = df_train[col].map(smooth).fillna(global_mean)
    df_test[f'{col}_TE'] = df_test[col].map(smooth).fillna(global_mean)
    cols_drop.append(col)

X = df_train.drop(columns=cols_drop, errors='ignore')
X_test_final = df_test.drop(columns=cols_drop, errors='ignore')

# Imputación
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test_final), columns=X_test_final.columns)

# ==========================================
# 4. FASE ACADÉMICA (CUMPLIMIENTO)
# ==========================================
print("\n" + "="*50)
print("   FASE 1: TESTEO DE MODELOS (REQUISITOS)")
print("="*50)

X_tr, X_val, y_tr, y_val = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# Requisito 1: Árbol
dt = DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42)
dt.fit(X_tr, y_tr)
print(f"   >> Árbol F1: {f1_score(y_val, dt.predict(X_val), average='macro'):.4f}")

# Requisito 2: Geométrico
geo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
geo.fit(X_tr, y_tr)
print(f"   >> Geométrico F1: {f1_score(y_val, geo.predict(X_val), average='macro'):.4f}")

# ==========================================
# 5. MODELO PRINCIPAL (ENSEMBLE V19)
# ==========================================
print("\n" + "="*50)
print("   FASE 2: ENTRENAMIENTO INICIAL")
print("="*50)

clf1 = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.015, max_depth=8, subsample=0.7, colsample_bytree=0.7, n_jobs=-1, random_state=42)
clf2 = RandomForestClassifier(n_estimators=500, max_depth=15, class_weight='balanced', n_jobs=-1, random_state=42)
clf3 = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, max_depth=10, l2_regularization=0.1, random_state=42)

ensemble = VotingClassifier(
    estimators=[('xgb', clf1), ('rf', clf2), ('hgb', clf3)],
    voting='soft',
    weights=[5, 3, 2]
)

# Entrenamiento 1 (Standard)
sample_weights = compute_sample_weight('balanced', y)
ensemble.fit(X_imputed, y, sample_weight=sample_weights)

# ==========================================
# 6. PSEUDO-LABELING (LA MAGIA PARA EL 0.80)
# ==========================================
print("\n" + "="*50)
print("   FASE 3: PSEUDO-LABELING (Refuerzo)")
print("="*50)

# 1. Predecir probabilidades del Test
probs_test = ensemble.predict_proba(X_test_imputed)

# 2. Filtrar predicciones MUY seguras (>99% o <1%)
# Cuanto más estricto sea el umbral, menos datos añadimos pero más seguros son.
CONFIDENCE_THRESHOLD = 0.95 

high_conf_indices = np.where((probs_test[:, 0] > CONFIDENCE_THRESHOLD) | (probs_test[:, 1] > CONFIDENCE_THRESHOLD))[0]
pseudo_X_test = X_test_imputed.iloc[high_conf_indices]
pseudo_y_test = (probs_test[high_conf_indices, 1] >= 0.5).astype(int)

print(f"   > Muestras originales de Train: {len(X_imputed)}")
print(f"   > Nuevas muestras 'seguras' del Test añadidas: {len(pseudo_X_test)}")

if len(pseudo_X_test) > 0:
    # 3. Añadir al Train
    X_aug = pd.concat([X_imputed, pseudo_X_test], axis=0)
    y_aug = pd.concat([y, pd.Series(pseudo_y_test)], axis=0)
    
    # Recalcular pesos para el nuevo set aumentado
    aug_weights = compute_sample_weight('balanced', y_aug)
    
    print("   > Re-entrenando Ensemble con datos aumentados...")
    
    # Re-definir el ensemble para resetearlo (limpio)
    # Importante: Usamos los mismos hiperparámetros
    final_ensemble = VotingClassifier(
        estimators=[('xgb', clf1), ('rf', clf2), ('hgb', clf3)],
        voting='soft',
        weights=[5, 3, 2]
    )
    
    final_ensemble.fit(X_aug, y_aug, sample_weight=aug_weights)
    model_for_prediction = final_ensemble
else:
    print("   > No hubo suficientes predicciones seguras. Usando modelo original.")
    model_for_prediction = ensemble

# ==========================================
# 7. PREDICCIÓN FINAL
# ==========================================
print("\nGenerando predicciones finales...")
final_preds = model_for_prediction.predict(X_test_imputed)

submission = pd.DataFrame({'id': test['id'], 'Accept': final_preds})
submission.to_csv(RUNS / 'submission_v21_pseudo_labeling.csv', index=False)
print("¡Archivo 'submission_v21_pseudo_labeling.csv' generado!")
print("Estrategia: V19 Base + Academic Check + Pseudo-Labeling (Data Augmentation).")