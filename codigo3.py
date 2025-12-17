import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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
# 2. LIMPIEZA AVANZADA
# ==========================================
print("--- 2. Limpieza y Feature Engineering ---")

# A. Moneda
def clean_currency(x):
    if isinstance(x, str):
        return float(x.replace('$', '').replace(',', '').strip())
    return x
for col in ['DisbursementGross', 'BalanceGross']:
    if col in df_all.columns: df_all[col] = df_all[col].apply(clean_currency)

# B. ApprovalFY
def clean_approval_fy(x):
    if pd.isna(x): return np.nan
    x = str(x).replace('A', '').strip() 
    try: return int(float(x)) 
    except: return np.nan
df_all['ApprovalFY'] = df_all['ApprovalFY'].apply(clean_approval_fy)

# C. Fechas y CORRECCIÓN DE AÑOS FUTUROS
date_cols = ['ApprovalDate', 'DisbursementDate']
for col in date_cols:
    df_all[col] = pd.to_datetime(df_all[col], errors='coerce')
    # Extraer año
    years = df_all[col].dt.year
    # Corregir años locos (ej: 2068 -> 1968)
    years = years.apply(lambda y: y - 100 if y > 2025 else y)
    df_all[f'{col}_Year'] = years
    df_all[f'{col}_Month'] = df_all[col].dt.month

df_all['Days_To_Disbursement'] = (df_all['DisbursementDate'] - df_all['ApprovalDate']).dt.days

# D. Categóricas y Valores Sucios (RevLineCr)
def clean_revline(x):
    if str(x) in ['Y', 'T', '1']: return 1
    if str(x) in ['N', '0']: return 0
    return np.nan # Dejar nulo para imputar
df_all['RevLineCr'] = df_all['RevLineCr'].apply(clean_revline)

def clean_lowdoc(x):
    if str(x) in ['Y', '1']: return 1
    if str(x) in ['N', '0', 'A', 'S']: return 0 # Asumimos raros como NO
    return np.nan
df_all['LowDoc'] = df_all['LowDoc'].apply(clean_lowdoc)

df_all['StateSame'] = (df_all['State'] == df_all['BankState']).astype(int)

# Recesión basada en año corregido
if 'DisbursementDate_Year' in df_all.columns:
    df_all['IsRecession'] = df_all['DisbursementDate_Year'].apply(lambda x: 1 if 2007 <= x <= 2009 else 0)

df_all['NoEmp'] = df_all['NoEmp'].replace(0, 1)
df_all['Amount_Per_Emp'] = df_all['DisbursementGross'] / df_all['NoEmp']

# ==========================================
# 3. ESTRATEGIA: AGRUPACIÓN & DROPS
# ==========================================
# Eliminar columna State si tiene 1 solo valor (detectado en análisis)
if df_all['State'].nunique() <= 1:
    print(">> Eliminando columna 'State' (Informatividad cero)")
    df_all.drop(columns=['State'], inplace=True, errors='ignore')
    cat_cols_low = ['NewExist', 'UrbanRural'] # Quitamos State de la lista
else:
    cat_cols_low = ['State', 'NewExist', 'UrbanRural']

def group_rare_labels(df, col, n_top=20):
    if col not in df.columns: return df, None
    top = df[col].value_counts().head(n_top).index.tolist()
    new_col = f'{col}_Grouped'
    df[new_col] = df[col].apply(lambda x: x if x in top else 'Other_Small')
    return df, new_col

df_all, bank_col_grouped = group_rare_labels(df_all, 'Bank', 15) 
df_all, city_col_grouped = group_rare_labels(df_all, 'City', 50)

# ==========================================
# 4. ENCODING
# ==========================================
cat_cols_high = [c for c in [bank_col_grouped, city_col_grouped, 'BankState', 'FranchiseCode'] if c]

le = LabelEncoder()
for col in cat_cols_low:
    df_all[col] = df_all[col].astype(str)
    df_all[col] = le.fit_transform(df_all[col])

cols_to_drop = ['id', 'LoanNr_ChkDgt', 'Name', 'ApprovalDate', 'DisbursementDate', 
                'is_train', 'Accept', 'Bank', 'City']

df_train = df_all[df_all['is_train'] == 1].copy()
df_test = df_all[df_all['is_train'] == 0].copy()
y = df_train['Accept'].astype(int)

# Target Encoding
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

imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test_final), columns=X_test_final.columns)
X_test_imputed['original_index'] = X_test_imputed.index

# ==========================================
# 5. MODELADO CON PESOS DE CLASE (BALANCEO)
# ==========================================
print("\n" + "="*50)
print("   ENTRENAMIENTO TIME-SPLIT BALANCEADO")
print("="*50)

# Calculamos pesos para corregir el desbalanceo
# Si Accept=0 es minoría, tendrá peso > 1. Si Accept=1 es mayoría, peso < 1.
# Esto obliga al modelo a prestar atención a los ceros.
sample_weights_global = compute_sample_weight(class_weight='balanced', y=y)

SPLIT_YEAR = 2006

# División Temporal
mask_pre = X_imputed['DisbursementDate_Year'] < SPLIT_YEAR
mask_post = X_imputed['DisbursementDate_Year'] >= SPLIT_YEAR

X_pre, y_pre = X_imputed[mask_pre], y[mask_pre]
X_post, y_post = X_imputed[mask_post], y[mask_post]

# Pesos específicos para cada época
sw_pre = compute_sample_weight('balanced', y_pre)
sw_post = compute_sample_weight('balanced', y_post)

# Modelos con soporte para weights
# Nota: XGBoost usa scale_pos_weight internamente o sample_weight en fit. Usaremos sample_weight en fit.
models_def = {
    'RF': RandomForestClassifier(n_estimators=300, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42),
    'XGB': xgb.XGBClassifier(n_estimators=600, learning_rate=0.02, max_depth=6, subsample=0.8, n_jobs=-1, random_state=42)
}

test_probs = {k: [] for k in models_def.keys()}
final_indices = []

# --- BUCLE DE ENTRENAMIENTO POR ÉPOCA ---
for era_name, X_era, y_era, sw_era, mask_test_era in [
    ('PRE-2008', X_pre, y_pre, sw_pre, X_test_imputed['DisbursementDate_Year'] < SPLIT_YEAR),
    ('POST-2008', X_post, y_post, sw_post, X_test_imputed['DisbursementDate_Year'] >= SPLIT_YEAR)
]:
    print(f"\n>> Procesando Era: {era_name} ({len(X_era)} muestras)")
    
    # Preparar test de esta era
    X_test_era = X_test_imputed[mask_test_era].copy()
    indices_era = X_test_era['original_index']
    final_indices.extend(indices_era)
    X_test_era = X_test_era.drop(columns=['original_index'])
    
    # Entrenar cada modelo
    era_probs = {}
    for m_name, model in models_def.items():
        # Clonar modelo nuevo
        m =  model.__class__(**model.get_params())
        
        # Entrenar con PESOS (sample_weight) es la clave
        m.fit(X_era, y_era, sample_weight=sw_era)
        
        # Predecir
        if len(X_test_era) > 0:
            probs = m.predict_proba(X_test_era)[:, 1]
            test_probs[m_name].extend(probs)
        else:
            print(f"   Advertencia: No hay datos de test para {era_name}")

# ==========================================
# 6. ENSEMBLE Y SUBMISSION
# ==========================================
print("\nGenerando Ensemble...")

# Reordenar probabilidades para que coincidan con el orden original
# Creamos un DF temporal para ordenar
df_probs = pd.DataFrame({'index': final_indices})
for m_name in models_def.keys():
    df_probs[m_name] = test_probs[m_name]

df_probs = df_probs.sort_values('index').set_index('index')

# Mezcla: RF suele ser muy bueno con clases desbalanceadas. XGBoost es potente.
# Damos 50/50 o 60/40. Probemos 50/50 para robustez.
final_prob = (0.5 * df_probs['XGB']) + (0.5 * df_probs['RF'])
final_pred = (final_prob >= 0.5).astype(int)

submission = pd.DataFrame({'id': test['id'], 'Accept': final_pred.values})
submission.to_csv('submission_balanced_final.csv', index=False)
print("¡Archivo 'submission_balanced_final.csv' generado!")
print("NOTA: Este modelo penaliza fuertemente fallar en los '0', lo que debería subir tu F1-Macro.")