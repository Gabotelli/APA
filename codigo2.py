import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
# 2. FEATURE ENGINEERING (SESGO INDUCTIVO)
# ==========================================
print("--- 2. Aplicando Sesgos Inductivos ---")

# A. Limpieza Estándar
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

# B. SESGO 1: TAMAÑO DE EMPRESA (Curva en U)
# Micro(1) -> Riskier(2-5) -> Safe(>5) -> VerySafe(>20)
df_all['NoEmp'] = df_all['NoEmp'].replace(0, 1)
df_all['CompanySize_Cat'] = pd.cut(df_all['NoEmp'], 
                                   bins=[-1, 1, 5, 20, 99999], 
                                   labels=['Micro', 'Small', 'Medium', 'Large'])

# C. SESGO 2: JERARQUÍA BANCARIA (Count Encoding)
# Los bancos que prestan mucho tienen políticas distintas a los que prestan poco
bank_counts = df_all['Bank'].value_counts()
df_all['Bank_Size'] = df_all['Bank'].map(bank_counts)

# D. Variables Categóricas y NLP
df_all['Name'] = df_all['Name'].fillna('').astype(str).str.upper()
df_all['Legal_Entity'] = df_all['Name'].apply(lambda x: 1 if any(w in x for w in ['LLC', 'INC', 'CORP', 'LTD']) else 0)
df_all['Sector_Risk'] = df_all['Name'].apply(lambda x: 1 if any(w in x for w in ['REALTY', 'ESTATE', 'CONST', 'BUILD']) else 0)

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
# Añadimos CompanySize_Cat al Label Encoding
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
# 4. CONFIGURACIÓN CON SESGO INDUCTIVO (MONOTONICIDAD)
# ==========================================
print("\n" + "="*50)
print("   ENTRENANDO CON MONOTONICIDAD FORZADA")
print("="*50)

# Detectar índice de columna para restricción
disbursement_col_idx = list(X.columns).index('DisbursementGross')
# Crear tupla de restricciones: 0=Nada, 1=Creciente
# Forzamos que DisbursementGross sea Creciente (Más dinero -> Más Aprobación)
monotone_constraints = '(' + ','.join(['1' if c == 'DisbursementGross' else '0' for c in X.columns]) + ')'

# XGBoost con Inductive Bias
xgb_clf = xgb.XGBClassifier(
    n_estimators=1000, learning_rate=0.015, max_depth=8, 
    subsample=0.7, colsample_bytree=0.7, n_jobs=-1, random_state=42,
    monotone_constraints=monotone_constraints # <--- AQUÍ ESTÁ LA MAGIA
)

# Random Forest (Estabilidad)
rf_clf = RandomForestClassifier(
    n_estimators=500, max_depth=15, class_weight='balanced', 
    n_jobs=-1, random_state=42
)

# Voting (Solo Expertos)
eclf = VotingClassifier(
    estimators=[('xgb', xgb_clf), ('rf', rf_clf)],
    voting='soft',
    weights=[6, 4]
)

# Pesos de Balanceo
sample_weights = compute_sample_weight('balanced', y)

print("Entrenando Ensemble Final...")
eclf.fit(X_imputed, y, sample_weight=sample_weights)

# ==========================================
# 5. SUBMISSION
# ==========================================
print("Generando predicciones...")
# Umbral ajustado manualmente a 0.45 (ligeramente conservador para capturar más 1s correctos)
final_probs = eclf.predict_proba(X_test_imputed)[:, 1]
final_preds = (final_probs >= 0.45).astype(int)

submission = pd.DataFrame({'id': test['id'], 'Accept': final_preds})
submission.to_csv('submission_v20_inductive_bias.csv', index=False)
print("¡Archivo 'submission_v20_inductive_bias.csv' generado!")
print(">> Sesgos aplicados: Monotonicidad en Dinero + Curva de Empresa + Tamaño Banco.")