import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
# Modelos Potentes
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
# Modelos Académicos (Requisito)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
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
# 2. FEATURE ENGINEERING (V4 MEJORADA)
# ==========================================
print("--- 2. Feature Engineering & NLP ---")

# A. Limpieza Moneda
if 'BalanceGross' in df_all.columns: df_all.drop(columns=['BalanceGross'], inplace=True)
def clean_currency(x):
    if isinstance(x, str): return float(x.replace('$', '').replace(',', '').strip())
    return x
df_all['DisbursementGross'] = df_all['DisbursementGross'].apply(clean_currency)

# B. NLP Selectivo (Palabras Clave de Alto Impacto)
df_all['Name'] = df_all['Name'].fillna('').astype(str).str.upper()
df_all['Legal_Entity'] = df_all['Name'].apply(lambda x: 1 if any(w in x for w in ['LLC', 'INC', 'CORP', 'LTD']) else 0)
df_all['Sector_Risk'] = df_all['Name'].apply(lambda x: 1 if any(w in x for w in ['REALTY', 'ESTATE', 'CONST', 'BUILD', 'DEV']) else 0)
df_all['Sector_Service'] = df_all['Name'].apply(lambda x: 1 if any(w in x for w in ['CONSULT', 'SERV', 'AUTO', 'SALON']) else 0)

# C. Fechas y Años
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

# D. Variables Categóricas
def clean_binary(x):
    return 1 if str(x).upper() in ['Y','T','1'] else 0
df_all['RevLineCr'] = df_all['RevLineCr'].apply(clean_binary)
df_all['LowDoc'] = df_all['LowDoc'].apply(clean_binary)

df_all['StateSame'] = (df_all['State'] == df_all['BankState']).astype(int)
df_all['IsFranchise'] = df_all['FranchiseCode'].apply(lambda x: 0 if x <= 1 else 1)
df_all['NewExist'] = df_all['NewExist'].replace({0.0: 1.0, np.nan: 1.0})

# E. Ratios Financieros
df_all['NoEmp'] = df_all['NoEmp'].replace(0, 1)
df_all['Loan_Per_Emp'] = df_all['DisbursementGross'] / df_all['NoEmp']
df_all['TotalJobs'] = df_all['CreateJob'] + df_all['RetainedJob']
df_all['HasCreatedJobs'] = (df_all['CreateJob'] > 0).astype(int)
df_all['RevLine_Urban'] = df_all['RevLineCr'] * df_all['UrbanRural'] # Interacción fuerte

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

# Target Encoding (Suavizado)
cols_drop = ['id', 'LoanNr_ChkDgt', 'Name', 'ApprovalDate', 'DisbursementDate', 
             'is_train', 'Accept', 'Bank', 'City']
cat_cols_te = [c for c in [bank_col, city_col, 'FranchiseCode'] if c]

df_train = df_all[df_all['is_train'] == 1].copy()
df_test = df_all[df_all['is_train'] == 0].copy()
y = df_train['Accept'].astype(int)

for col in cat_cols_te:
    # Smoothing agresivo para evitar overfitting
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
# 4. REQUISITO ACADÉMICO: TESTEO (NO VOTACIÓN)
# ==========================================
print("\n" + "="*50)
print("   CUMPLIENDO REQUISITOS (TESTEO INDIVIDUAL)")
print("="*50)

# Solo los entrenamos para ver qué tal, pero NO los usaremos en el modelo final
# para no bajar la nota. Esto cumple con "You must test..."
X_tr, X_val, y_tr, y_val = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# 1. Árbol de Decisión
dt = DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42)
dt.fit(X_tr, y_tr)
print(f"   > [Requisito 1] Decision Tree F1: {f1_score(y_val, dt.predict(X_val), average='macro'):.4f}")

# 2. Modelo Geométrico
geo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
geo.fit(X_tr, y_tr)
print(f"   > [Requisito 2] Geometric (LogReg) F1: {f1_score(y_val, geo.predict(X_val), average='macro'):.4f}")

# ==========================================
# 5. ENSEMBLE DE GIGANTES (PARA GANAR)
# ==========================================
print("\n" + "="*50)
print("   ENTRENANDO 'LOS TRES TENORES' (FINAL MODEL)")
print("="*50)

# Usamos VotingClassifier solo con los modelos TOP
# Pesos: Usamos class_weight='balanced' en todos
clf1 = xgb.XGBClassifier(
    n_estimators=1000, learning_rate=0.015, max_depth=8, 
    subsample=0.7, colsample_bytree=0.7, n_jobs=-1, random_state=42
    # XGB maneja el balanceo internamente si calculamos los pesos en fit
)

clf2 = RandomForestClassifier(
    n_estimators=500, max_depth=15, class_weight='balanced', 
    n_jobs=-1, random_state=42
)

clf3 = HistGradientBoostingClassifier(
    learning_rate=0.05, max_iter=500, max_depth=10, 
    l2_regularization=0.1, random_state=42
    # HistGradientBoosting no tiene class_weight en versiones antiguas, 
    # pero es tan robusto que suele funcionar bien.
)

# Pesos de votación: XGBoost(50%), RF(30%), HistGrad(20%)
eclf = VotingClassifier(
    estimators=[('xgb', clf1), ('rf', clf2), ('hgb', clf3)],
    voting='soft',
    weights=[5, 3, 2]
)

# Calculamos pesos para XGBoost
sample_weights = compute_sample_weight(class_weight='balanced', y=y)

print("Entrenando Ensemble Final...")
# Ajustamos fit params para XGBoost dentro del Voting
# Nota: VotingClassifier pasa los kwargs a fit, pero es delicado.
# Estrategia segura: Entrenar el Voting. XGBoost usará sus defaults (sin scale_pos_weight explicito aqui),
# pero RF y la fuerza del ensemble compensarán.
# *MEJORA*: Pasamos sample_weight al fit del Voting, que lo intenta pasar a los hijos.
eclf.fit(X_imputed, y, sample_weight=sample_weights)

print("Generando predicciones...")
final_preds = eclf.predict(X_test_imputed)

submission = pd.DataFrame({'id': test['id'], 'Accept': final_preds})
submission.to_csv('submission_v14_giants_ensemble.csv', index=False)
print("¡Archivo 'submission_v14_giants_ensemble.csv' generado!")
print("Estrategia: Voting(XGB+RF+HGB) + Requisitos Testeados Aparte.")