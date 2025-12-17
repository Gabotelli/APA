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
# 2. FEATURE ENGINEERING (LA FÓRMULA DEL 0.76)
# ==========================================
print("--- 2. Feature Engineering (V14 Restaurada) ---")

# A. Limpieza Básica
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

# B. NLP Selectivo (Factor de Riesgo)
df_all['Name'] = df_all['Name'].fillna('').astype(str).str.upper()
df_all['Legal_Entity'] = df_all['Name'].apply(lambda x: 1 if any(w in x for w in ['LLC', 'INC', 'CORP', 'LTD']) else 0)
df_all['Sector_Risk'] = df_all['Name'].apply(lambda x: 1 if any(w in x for w in ['REALTY', 'ESTATE', 'CONST', 'BUILD', 'DEV']) else 0)
df_all['Sector_Service'] = df_all['Name'].apply(lambda x: 1 if any(w in x for w in ['CONSULT', 'SERV', 'AUTO', 'SALON']) else 0)

# C. Variables Clave
def clean_binary(x):
    return 1 if str(x).upper() in ['Y','T','1'] else 0
df_all['RevLineCr'] = df_all['RevLineCr'].apply(clean_binary)
df_all['LowDoc'] = df_all['LowDoc'].apply(clean_binary)

df_all['StateSame'] = (df_all['State'] == df_all['BankState']).astype(int)
df_all['IsFranchise'] = df_all['FranchiseCode'].apply(lambda x: 0 if x <= 1 else 1)
df_all['NewExist'] = df_all['NewExist'].replace({0.0: 1.0, np.nan: 1.0})

df_all['NoEmp'] = df_all['NoEmp'].replace(0, 1)
df_all['Loan_Per_Emp'] = df_all['DisbursementGross'] / df_all['NoEmp']
df_all['TotalJobs'] = df_all['CreateJob'] + df_all['RetainedJob']
df_all['HasCreatedJobs'] = (df_all['CreateJob'] > 0).astype(int)
# Esta interacción subió mucho la nota en V14
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

# Target Encoding Agresivo
cols_drop = ['id', 'LoanNr_ChkDgt', 'Name', 'ApprovalDate', 'DisbursementDate', 
             'is_train', 'Accept', 'Bank', 'City']
cat_cols_te = [c for c in [bank_col, city_col, 'FranchiseCode'] if c]

df_train = df_all[df_all['is_train'] == 1].copy()
df_test = df_all[df_all['is_train'] == 0].copy()
y = df_train['Accept'].astype(int)

for col in cat_cols_te:
    global_mean = y.mean()
    agg = df_train.groupby(col)['Accept'].agg(['count', 'mean'])
    # Smoothing +20 funcionó mejor en V14
    smooth = (agg['count'] * agg['mean'] + 20 * global_mean) / (agg['count'] + 20)
    df_train[f'{col}_TE'] = df_train[col].map(smooth).fillna(global_mean)
    df_test[f'{col}_TE'] = df_test[col].map(smooth).fillna(global_mean)
    cols_drop.append(col)

X = df_train.drop(columns=cols_drop, errors='ignore')
X_test_final = df_test.drop(columns=cols_drop, errors='ignore')

imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test_final), columns=X_test_final.columns)

# ==========================================
# 4. FASE 1: TESTEO DE REQUISITOS (SOLO INFORME)
# ==========================================
print("\n" + "="*50)
print("   FASE ACADÉMICA: Testeo de Modelos Básicos")
print("   (Estos resultados van al Notebook, pero NO al archivo final)")
print("="*50)

X_tr, X_val, y_tr, y_val = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# Requisito 1: Árbol de Decisión
dt = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)
dt.fit(X_tr, y_tr)
f1_dt = f1_score(y_val, dt.predict(X_val), average='macro')
print(f"   >> ÁRBOL DE DECISIÓN F1: {f1_dt:.4f}")

# Requisito 2: Geométrico (Regresión Logística)
geo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
geo.fit(X_tr, y_tr)
f1_geo = f1_score(y_val, geo.predict(X_val), average='macro')
print(f"   >> GEOMÉTRICO (LogReg) F1: {f1_geo:.4f}")

# ==========================================
# 5. FASE 2: MODELO DE COMPETICIÓN (V14 PURA)
# ==========================================
print("\n" + "="*50)
print("   FASE COMPETICIÓN: Ensemble 'Los Tres Tenores'")
print("   (Recuperando la estrategia del 0.763)")
print("="*50)

# Definimos SOLO los modelos potentes para el Voting
# 1. XGBoost
clf1 = xgb.XGBClassifier(
    n_estimators=1000, learning_rate=0.015, max_depth=8, 
    subsample=0.7, colsample_bytree=0.7, n_jobs=-1, random_state=42
)

# 2. Random Forest
clf2 = RandomForestClassifier(
    n_estimators=500, max_depth=15, class_weight='balanced', 
    n_jobs=-1, random_state=42
)

# 3. HistGradientBoosting
clf3 = HistGradientBoostingClassifier(
    learning_rate=0.05, max_iter=500, max_depth=10, 
    l2_regularization=0.1, random_state=42
    # En V14 usamos este y funcionó genial sin weights explicitos
)

# Voting SIN los modelos académicos (para no bajar la media)
eclf = VotingClassifier(
    estimators=[('xgb', clf1), ('rf', clf2), ('hgb', clf3)],
    voting='soft',
    weights=[5, 3, 2] # XGBoost domina
)

# Pesos de muestra (EL SECRETO QUE FALTÓ EN V18)
sample_weights = compute_sample_weight('balanced', y)

print("Entrenando Modelo Final con Pesos de Balanceo...")
# Al NO incluir pipelines en este Voting, podemos pasar sample_weight sin error
eclf.fit(X_imputed, y, sample_weight=sample_weights)

print("Generando predicciones...")
final_preds = eclf.predict(X_test_imputed)

submission = pd.DataFrame({'id': test['id'], 'Accept': final_preds})
submission.to_csv('submission_v19_restoration.csv', index=False)
print("¡Archivo 'submission_v19_restoration.csv' generado!")
print(">> Estrategia: Voting (XGB+RF+HGB) + Balanceo correcto.")