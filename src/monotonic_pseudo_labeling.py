import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
# 2. FEATURE ENGINEERING (LA BASE V20 - INDUCTIVE)
# ==========================================
print("--- 2. Feature Engineering (V20 Inductive) ---")

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

# A. SESGO: TAMAÑO DE EMPRESA (Curva en U)
df_all['NoEmp'] = df_all['NoEmp'].replace(0, 1)
df_all['CompanySize_Cat'] = pd.cut(df_all['NoEmp'], 
                                   bins=[-1, 1, 5, 20, 99999], 
                                   labels=['Micro', 'Small', 'Medium', 'Large'])

# B. SESGO: JERARQUÍA BANCARIA
bank_counts = df_all['Bank'].value_counts()
df_all['Bank_Size'] = df_all['Bank'].map(bank_counts)

# C. NLP Selectivo
df_all['Name'] = df_all['Name'].fillna('').astype(str).str.upper()
df_all['Legal_Entity'] = df_all['Name'].apply(lambda x: 1 if any(w in x for w in ['LLC', 'INC', 'CORP', 'LTD']) else 0)
df_all['Sector_Risk'] = df_all['Name'].apply(lambda x: 1 if any(w in x for w in ['REALTY', 'ESTATE', 'CONST', 'BUILD']) else 0)

# Categóricas
def clean_binary(x):
    return 1 if str(x).upper() in ['Y','T','1'] else 0
df_all['RevLineCr'] = df_all['RevLineCr'].apply(clean_binary)
df_all['LowDoc'] = df_all['LowDoc'].apply(clean_binary)

df_all['StateSame'] = (df_all['State'] == df_all['BankState']).astype(int)
df_all['IsFranchise'] = df_all['FranchiseCode'].apply(lambda x: 0 if x <= 1 else 1)
df_all['NewExist'] = df_all['NewExist'].replace({0.0: 1.0, np.nan: 1.0})

# Ratios
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
# Incluimos CompanySize_Cat en el Label Encoding
for col in ['NewExist', 'UrbanRural', 'BankState', 'CompanySize_Cat']:
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
# 4. CONFIGURACIÓN DEL MODELO (V20 CONSTRAINTS)
# ==========================================
print("\n" + "="*50)
print("   CONFIGURANDO ENSEMBLE CON SESGOS")
print("="*50)

# Monotonicidad forzada para XGBoost (Más dinero -> Más seguro)
try:
    disbursement_col_idx = list(X_imputed.columns).index('DisbursementGross')
    monotone_constraints = '(' + ','.join(['1' if c == 'DisbursementGross' else '0' for c in X_imputed.columns]) + ')'
except:
    monotone_constraints = None

clf_xgb = xgb.XGBClassifier(
    n_estimators=1000, learning_rate=0.015, max_depth=8, 
    subsample=0.7, colsample_bytree=0.7, n_jobs=-1, random_state=42,
    monotone_constraints=monotone_constraints # Sesgo Inductivo
)

clf_rf = RandomForestClassifier(
    n_estimators=500, max_depth=15, class_weight='balanced', 
    n_jobs=-1, random_state=42
)

ensemble = VotingClassifier(
    estimators=[('xgb', clf_xgb), ('rf', clf_rf)],
    voting='soft',
    weights=[6, 4]
)

# ==========================================
# 5. PSEUDO-LABELING (LA MAGIA DE V21)
# ==========================================
print("\n" + "="*50)
print("   FASE 1: ENTRENAMIENTO INICIAL")
print("="*50)

sample_weights = compute_sample_weight('balanced', y)
ensemble.fit(X_imputed, y, sample_weight=sample_weights)

print("\n" + "="*50)
print("   FASE 2: PSEUDO-LABELING (FUSIÓN)")
print("="*50)

# 1. Predecir Test
probs_test = ensemble.predict_proba(X_test_imputed)

# 2. Filtrar predicciones MUY seguras (Umbral estricto 97%)
CONFIDENCE_THRESHOLD = 0.97
high_conf_indices = np.where((probs_test[:, 0] > CONFIDENCE_THRESHOLD) | (probs_test[:, 1] > CONFIDENCE_THRESHOLD))[0]

pseudo_X_test = X_test_imputed.iloc[high_conf_indices]
pseudo_y_test = (probs_test[high_conf_indices, 1] >= 0.5).astype(int)

print(f"   > Muestras Train originales: {len(X_imputed)}")
print(f"   > Nuevas muestras seguras (Test): {len(pseudo_X_test)}")

if len(pseudo_X_test) > 0:
    # 3. Añadir al Train
    X_aug = pd.concat([X_imputed, pseudo_X_test], axis=0)
    y_aug = pd.concat([y, pd.Series(pseudo_y_test)], axis=0)
    
    # Recalcular pesos
    aug_weights = compute_sample_weight('balanced', y_aug)
    
    print("   > Re-entrenando Ensemble inteligente...")
    
    # Reiniciar modelos
    final_ensemble = VotingClassifier(
        estimators=[('xgb', clf_xgb), ('rf', clf_rf)],
        voting='soft',
        weights=[6, 4]
    )
    
    final_ensemble.fit(X_aug, y_aug, sample_weight=aug_weights)
    model_prediction = final_ensemble
else:
    print("   > No suficientes datos seguros. Usando modelo original.")
    model_prediction = ensemble

# ==========================================
# 6. PREDICCIÓN FINAL
# ==========================================
print("\nGenerando predicciones finales...")
# Umbral 0.45 del V20
final_probs = model_prediction.predict_proba(X_test_imputed)[:, 1]
final_preds = (final_probs >= 0.45).astype(int)

submission = pd.DataFrame({'id': test['id'], 'Accept': final_preds})
submission.to_csv(RUNS / 'submission_v22_fusion_final.csv', index=False)
print("¡Archivo 'submission_v22_fusion_final.csv' generado!")