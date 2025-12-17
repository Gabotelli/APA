import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

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
# 2. LIMPIEZA Y FEATURE ENGINEERING
# ==========================================
print("--- 2. Limpieza y Feature Engineering ---")

def clean_currency(x):
    if isinstance(x, str):
        return float(x.replace('$', '').replace(',', '').strip())
    return x

for col in ['DisbursementGross', 'BalanceGross']:
    if col in df_all.columns:
        df_all[col] = df_all[col].apply(clean_currency)

def clean_approval_fy(x):
    if pd.isna(x): return np.nan
    x = str(x).replace('A', '').strip() 
    try: return int(float(x)) 
    except: return np.nan

df_all['ApprovalFY'] = df_all['ApprovalFY'].apply(clean_approval_fy)

date_cols = ['ApprovalDate', 'DisbursementDate']
for col in date_cols:
    df_all[col] = pd.to_datetime(df_all[col], errors='coerce')
    df_all[f'{col}_Year'] = df_all[col].dt.year
    df_all[f'{col}_Month'] = df_all[col].dt.month

df_all['Days_To_Disbursement'] = (df_all['DisbursementDate'] - df_all['ApprovalDate']).dt.days

def clean_binary(x):
    if str(x).upper() in ['Y', 'YES', '1']: return 1
    elif str(x).upper() in ['N', 'NO', '0']: return 0
    else: return np.nan 

df_all['RevLineCr'] = df_all['RevLineCr'].apply(clean_binary)
df_all['LowDoc'] = df_all['LowDoc'].apply(clean_binary)
df_all['StateSame'] = (df_all['State'] == df_all['BankState']).astype(int)

if 'DisbursementDate_Year' in df_all.columns:
    df_all['IsRecession'] = df_all['DisbursementDate_Year'].apply(lambda x: 1 if 2007 <= x <= 2009 else 0)

df_all['NoEmp'] = df_all['NoEmp'].replace(0, 1)
df_all['Amount_Per_Emp'] = df_all['DisbursementGross'] / df_all['NoEmp']

# ==========================================
# 3. ESTRATEGIA: AGRUPACIÓN DE BANCOS
# ==========================================
print("--- 3. Agrupando Bancos Pequeños ---")

def group_rare_labels(df, col, n_top=20):
    top_banks = df[col].value_counts().head(n_top).index.tolist()
    new_col_name = f'{col}_Grouped'
    df[new_col_name] = df[col].apply(lambda x: x if x in top_banks else 'Other_Small')
    return df, new_col_name

df_all, bank_col_grouped = group_rare_labels(df_all, 'Bank', n_top=25) 
df_all, city_col_grouped = group_rare_labels(df_all, 'City', n_top=50)

# ==========================================
# 4. ENCODING
# ==========================================
print("--- 4. Encoding ---")

cat_cols_high = [bank_col_grouped, city_col_grouped, 'BankState', 'FranchiseCode'] 
cat_cols_low = ['State', 'NewExist', 'UrbanRural'] 

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

# Imputación (manteniendo DataFrame para facilitar filtrado por año)
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test_final), columns=X_test_final.columns)

# Guardamos índices de test para reconstruir orden
X_test_imputed['original_index'] = X_test_imputed.index

# ==========================================
# 5. DEFINICIÓN DE MODELOS
# ==========================================
SPLIT_YEAR = 2008

def get_models():
    # Devuelve un diccionario con instancias nuevas de los modelos
    return {
        'LogisticRegression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        'DecisionTree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, eval_metric='logloss')
    }

# ==========================================
# 6. ENTRENAMIENTO Y VALIDACIÓN (TIME-SPLIT)
# ==========================================
print("\n" + "="*50)
print("   ENTRENAMIENTO TIME-SPLIT (PRE vs POST 2008)")
print("="*50)

# Diccionario para guardar predicciones de TEST (probabilidades) de cada modelo
test_probs = {model_name: pd.Series(index=X_test_imputed.index, dtype=float) for model_name in get_models().keys()}
# Diccionario para guardar scores de validación
val_scores = {}

# --- A. Dividir Datos en Épocas ---
train_mask_pre = X_imputed['DisbursementDate_Year'] < SPLIT_YEAR
train_mask_post = X_imputed['DisbursementDate_Year'] >= SPLIT_YEAR

X_pre = X_imputed[train_mask_pre]
y_pre = y[train_mask_pre]
X_post = X_imputed[train_mask_post]
y_post = y[train_mask_post]

test_mask_pre = X_test_imputed['DisbursementDate_Year'] < SPLIT_YEAR
test_mask_post = X_test_imputed['DisbursementDate_Year'] >= SPLIT_YEAR

# Separar Test y guardar índices
X_test_pre = X_test_imputed[test_mask_pre].copy()
idx_test_pre = X_test_pre['original_index']
X_test_pre = X_test_pre.drop(columns=['original_index'])

X_test_post = X_test_imputed[test_mask_post].copy()
idx_test_post = X_test_post['original_index']
X_test_post = X_test_post.drop(columns=['original_index'])

# --- B. Bucle por Modelo ---
models_dict = get_models() # Instancias base

for name, _ in models_dict.items():
    print(f"\nProcesando Modelo: {name}...")
    
    # 1. Validación Interna (Para saber qué tan bueno es)
    # ---------------------------------------------------
    # Split Pre
    tr_pre, val_pre, y_tr_pre, y_val_pre = train_test_split(X_pre, y_pre, test_size=0.2, random_state=42)
    # Split Post
    tr_post, val_post, y_tr_post, y_val_post = train_test_split(X_post, y_post, test_size=0.2, random_state=42)
    
    # Instancias específicas
    m_pre = get_models()[name] # Nueva instancia
    m_post = get_models()[name] # Nueva instancia
    
    # Entrenar en Train parcial
    m_pre.fit(tr_pre, y_tr_pre)
    m_post.fit(tr_post, y_tr_post)
    
    # Evaluar
    pred_val_pre = m_pre.predict(val_pre)
    pred_val_post = m_post.predict(val_post)
    
    f1_pre = f1_score(y_val_pre, pred_val_pre, average='macro')
    f1_post = f1_score(y_val_post, pred_val_post, average='macro')
    
    # Promedio ponderado
    weighted_f1 = (f1_pre * len(val_pre) + f1_post * len(val_post)) / (len(val_pre) + len(val_post))
    val_scores[name] = weighted_f1
    print(f"   > F1 Pre-2008: {f1_pre:.4f} | F1 Post-2008: {f1_post:.4f} | Global Avg: {weighted_f1:.4f}")
    
    # 2. Entrenamiento Final y Predicción TEST
    # ----------------------------------------
    # Re-entrenamos con TODO el dato de la época (Train + Val) para máxima potencia
    m_final_pre = get_models()[name]
    m_final_pre.fit(X_pre, y_pre)
    
    m_final_post = get_models()[name]
    m_final_post.fit(X_post, y_post)
    
    # Predecir Probabilidades en Test
    probs_pre = m_final_pre.predict_proba(X_test_pre)[:, 1]
    probs_post = m_final_post.predict_proba(X_test_post)[:, 1]
    
    # Reconstruir el vector completo de probabilidades
    s_pre = pd.Series(probs_pre, index=idx_test_pre)
    s_post = pd.Series(probs_post, index=idx_test_post)
    s_total = pd.concat([s_pre, s_post]).sort_index()
    
    test_probs[name] = s_total

# ==========================================
# 7. ENSEMBLE (MEZCLA DE EXPERTOS)
# ==========================================
print("\n" + "="*50)
print("   GENERANDO ENSEMBLE FINAL")
print("="*50)

# Definimos pesos basados en el rendimiento esperado (XGBoost y RF suelen mandar)
# Puedes ajustar esto según los 'Global Avg' que veas en la consola
weights = {
    'XGBoost': 0.50,
    'RandomForest': 0.30,
    'DecisionTree': 0.15,
    'LogisticRegression': 0.05
}

print("Pesos del Ensemble:", weights)

final_blend_probs = np.zeros(len(test))

for name, weight in weights.items():
    final_blend_probs += test_probs[name].values * weight

# Convertir a clase (0 o 1)
final_blend_preds = (final_blend_probs >= 0.5).astype(int)

# ==========================================
# 8. GENERAR ARCHIVOS
# ==========================================

# A. Submission del Ensemble
sub_ensemble = pd.DataFrame({'id': test['id'], 'Accept': final_blend_preds})
sub_ensemble.to_csv('submission_ensemble_timesplit.csv', index=False)
print("\n>> Archivo 'submission_ensemble_timesplit.csv' generado (Tu mejor apuesta).")

# B. Submission del Mejor Individual (Por si acaso el ensemble falla)
best_model_name = max(val_scores, key=val_scores.get)
best_model_preds = (test_probs[best_model_name].values >= 0.5).astype(int)

sub_best = pd.DataFrame({'id': test['id'], 'Accept': best_model_preds})
sub_best.to_csv(f'submission_best_{best_model_name}_timesplit.csv', index=False)
print(f">> Archivo 'submission_best_{best_model_name}_timesplit.csv' generado (Plan B).")

print("\n¡Proceso Finalizado con Éxito! 🚀")