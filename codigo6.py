import pandas as pd
import numpy as np
import time
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
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
# 2. FEATURE ENGINEERING (CORREGIDO)
# ==========================================
print("--- 2. Limpieza, NLP y Correcciones ---")

# A. Moneda
def clean_currency(x):
    if isinstance(x, str): return float(x.replace('$', '').replace(',', '').strip())
    return x
for col in ['DisbursementGross', 'BalanceGross']:
    df_all[col] = df_all[col].apply(clean_currency)

# B. CORRECCIÓN DEL ERROR '1976A' (ApprovalFY)
# ---------------------------------------------------------
def clean_approval_fy(x):
    if pd.isna(x): return np.nan
    x = str(x).replace('A', '').strip() 
    try: return int(float(x)) 
    except: return np.nan

df_all['ApprovalFY'] = df_all['ApprovalFY'].apply(clean_approval_fy)
# ---------------------------------------------------------

# C. NLP en Nombres (Estructura y Sector)
df_all['Name'] = df_all['Name'].fillna('').astype(str).str.upper()
df_all['Legal_LLC'] = df_all['Name'].apply(lambda x: 1 if 'LLC' in x else 0)
df_all['Legal_INC'] = df_all['Name'].apply(lambda x: 1 if 'INC' in x or 'CORP' in x else 0)
df_all['Sector_RealEstate'] = df_all['Name'].apply(lambda x: 1 if 'REALTY' in x or 'ESTATE' in x else 0)

# D. Fechas y Años
for col in ['ApprovalDate', 'DisbursementDate']:
    df_all[col] = pd.to_datetime(df_all[col], errors='coerce')
    years = df_all[col].dt.year
    # Corregir años futuros
    years = years.apply(lambda y: y - 100 if y > 2025 else y)
    df_all[f'{col}_Year'] = years

df_all['Days_To_Disbursement'] = (df_all['DisbursementDate'] - df_all['ApprovalDate']).dt.days

# E. Limpieza Categórica y Binaria
df_all['RevLineCr'] = df_all['RevLineCr'].apply(lambda x: 1 if str(x) in ['Y','T','1'] else 0)
df_all['LowDoc'] = df_all['LowDoc'].apply(lambda x: 1 if str(x) in ['Y','1'] else 0)
df_all['StateSame'] = (df_all['State'] == df_all['BankState']).astype(int)
df_all['IsFranchise'] = df_all['FranchiseCode'].apply(lambda x: 0 if x <= 1 else 1)

# F. Ratios
df_all['NoEmp'] = df_all['NoEmp'].replace(0, 1)
df_all['Loan_Per_Emp'] = df_all['DisbursementGross'] / df_all['NoEmp']

# Drop y Grouping
if df_all['State'].nunique() <= 1: df_all.drop(columns=['State'], inplace=True, errors='ignore')

def group_rare(df, col, n=25):
    top = df[col].value_counts().head(n).index.tolist()
    df[f'{col}_Grouped'] = df[col].apply(lambda x: x if x in top else 'Other')
    return df, f'{col}_Grouped'

df_all, bank_col = group_rare(df_all, 'Bank', 30)
df_all, city_col = group_rare(df_all, 'City', 60)

# Label Encoding
cat_cols_simple = ['NewExist', 'UrbanRural', 'BankState']
le = LabelEncoder()
for col in cat_cols_simple:
    df_all[col] = df_all[col].astype(str)
    df_all[col] = le.fit_transform(df_all[col])

# Target Encoding
cols_to_drop = ['id', 'LoanNr_ChkDgt', 'Name', 'ApprovalDate', 'DisbursementDate', 
                'is_train', 'Accept', 'Bank', 'City', 'BalanceGross']
cat_cols_te = [bank_col, city_col, 'FranchiseCode']

df_train = df_all[df_all['is_train'] == 1].copy()
df_test = df_all[df_all['is_train'] == 0].copy()
y = df_train['Accept'].astype(int)

for col in cat_cols_te:
    mapping = df_train.groupby(col)['Accept'].mean()
    df_train[f'{col}_TE'] = df_train[col].map(mapping).fillna(y.mean())
    df_test[f'{col}_TE'] = df_test[col].map(mapping).fillna(y.mean())
    cols_to_drop.append(col)

X = df_train.drop(columns=cols_to_drop, errors='ignore')
X_test_final = df_test.drop(columns=cols_to_drop, errors='ignore')

# Imputación (Ahora sí funcionará porque ApprovalFY es numérico)
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test_final), columns=X_test_final.columns)
X_test_imputed['original_index'] = X_test_imputed.index

# ==========================================
# 3. DEFINICIÓN DE MODELOS (CORREGIDO)
# ==========================================
print("\n" + "="*50)
print("   ENTRENANDO ENSEMBLE 'ALL-STARS' (V6)")
print("="*50)

models = {
    'XGB': xgb.XGBClassifier(n_estimators=600, learning_rate=0.02, max_depth=7, subsample=0.7, n_jobs=-1, random_state=42),
    
    'RF': RandomForestClassifier(n_estimators=300, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42),
    
    # CORRECCIÓN AQUÍ: Quitamos "algorithm='SAMME'"
    'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    
    'SVM': make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, probability=True, cache_size=1000, random_state=42)),
    
    'LogReg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
}

SPLIT_YEAR = 2008
test_probs = {k: [] for k in models.keys()}
final_indices = []

# ==========================================
# 4. ENTRENAMIENTO POR ÉPOCAS (TIME-SPLIT)
# ==========================================
for era_name, mask_tr, mask_te in [
    ('PRE-CRISIS', X_imputed['DisbursementDate_Year'] < SPLIT_YEAR, X_test_imputed['DisbursementDate_Year'] < SPLIT_YEAR),
    ('POST-CRISIS', X_imputed['DisbursementDate_Year'] >= SPLIT_YEAR, X_test_imputed['DisbursementDate_Year'] >= SPLIT_YEAR)
]:
    print(f"\n>>> Procesando Era: {era_name}")
    X_era = X_imputed[mask_tr]
    y_era = y[mask_tr]
    
    X_te_era = X_test_imputed[mask_te].copy()
    idxs = X_te_era['original_index']
    final_indices.extend(idxs)
    X_te_era = X_te_era.drop(columns=['original_index'])
    
    # Pesos
    sw = compute_sample_weight('balanced', y_era)
    
    for name, model in models.items():
        print(f"   Entrenando {name}...")
        
        # Ajuste de pesos según soporte del modelo
        if name in ['XGB', 'RF']:
            model.fit(X_era, y_era, sample_weight=sw)
        elif name == 'AdaBoost':
             model.fit(X_era, y_era, sample_weight=sw)
        else:
            model.fit(X_era, y_era) # SVM y LogReg usan class_weight interno
            
        # Predecir
        if len(X_te_era) > 0:
            probs = model.predict_proba(X_te_era)[:, 1]
            test_probs[name].extend(probs)
        else:
            print(f"     (Sin datos de test para {name})")

# ==========================================
# 5. ENSEMBLE VOTING
# ==========================================
print("\nGenerando Votación Final...")

df_probs = pd.DataFrame({'index': final_indices})
for name in models.keys():
    df_probs[name] = test_probs[name]
df_probs = df_probs.sort_values('index').set_index('index')

# PESOS ESTRATÉGICOS (Diversidad)
final_prob = (0.40 * df_probs['XGB']) + \
             (0.25 * df_probs['RF']) + \
             (0.15 * df_probs['SVM']) + \
             (0.15 * df_probs['AdaBoost']) + \
             (0.05 * df_probs['LogReg'])

final_pred = (final_prob >= 0.5).astype(int)

submission = pd.DataFrame({'id': test['id'], 'Accept': final_pred.values})
submission.to_csv('submission_v6_allstars.csv', index=False)
print("¡Archivo 'submission_v6_allstars.csv' generado con éxito!")