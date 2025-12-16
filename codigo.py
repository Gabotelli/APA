import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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

# A. Moneda
def clean_currency(x):
    if isinstance(x, str):
        return float(x.replace('$', '').replace(',', '').strip())
    return x

for col in ['DisbursementGross', 'BalanceGross']:
    if col in df_all.columns:
        df_all[col] = df_all[col].apply(clean_currency)

# B. ApprovalFY
def clean_approval_fy(x):
    if pd.isna(x): return np.nan
    x = str(x).replace('A', '').strip() 
    try: return int(float(x)) 
    except: return np.nan

df_all['ApprovalFY'] = df_all['ApprovalFY'].apply(clean_approval_fy)

# C. Fechas y Duración
date_cols = ['ApprovalDate', 'DisbursementDate']
for col in date_cols:
    df_all[col] = pd.to_datetime(df_all[col], errors='coerce')
    df_all[f'{col}_Year'] = df_all[col].dt.year
    df_all[f'{col}_Month'] = df_all[col].dt.month

# Feature: Días hasta desembolso
df_all['Days_To_Disbursement'] = (df_all['DisbursementDate'] - df_all['ApprovalDate']).dt.days

# D. Categóricas Binarias
def clean_binary(x):
    if str(x).upper() in ['Y', 'YES', '1']: return 1
    elif str(x).upper() in ['N', 'NO', '0']: return 0
    else: return np.nan 

df_all['RevLineCr'] = df_all['RevLineCr'].apply(clean_binary)
df_all['LowDoc'] = df_all['LowDoc'].apply(clean_binary)

# Feature: Misma Entidad
df_all['StateSame'] = (df_all['State'] == df_all['BankState']).astype(int)

# Feature: Recesión
if 'DisbursementDate_Year' in df_all.columns:
    df_all['IsRecession'] = df_all['DisbursementDate_Year'].apply(lambda x: 1 if 2007 <= x <= 2009 else 0)

# Feature: Monto por Empleado
df_all['NoEmp'] = df_all['NoEmp'].replace(0, 1)
df_all['Amount_Per_Emp'] = df_all['DisbursementGross'] / df_all['NoEmp']

# ==========================================
# 3. ESTRATEGIA: AGRUPACIÓN DE BANCOS (Fixed)
# ==========================================
print("--- 3. Agrupando Bancos y Ciudades Pequeñas ---")

def group_rare_labels(df, col, n_top=20):
    top_banks = df[col].value_counts().head(n_top).index.tolist()
    # Nueva columna con sufijo _Grouped
    new_col_name = f'{col}_Grouped'
    df[new_col_name] = df[col].apply(lambda x: x if x in top_banks else 'Other_Small')
    print(f"   > Columna '{new_col_name}' creada. Top {n_top} mantenidos, resto agrupados.")
    return df, new_col_name

# Aplicamos la estrategia
df_all, bank_col_grouped = group_rare_labels(df_all, 'Bank', n_top=25) 
df_all, city_col_grouped = group_rare_labels(df_all, 'City', n_top=50)

# ==========================================
# 4. ENCODING AVANZADO
# ==========================================
print("--- 4. Encoding (Label + Target) ---")

# AQUÍ ESTABA EL ERROR: Ahora usamos las columnas _Grouped, no las originales
cat_cols_high = [bank_col_grouped, city_col_grouped, 'BankState', 'FranchiseCode'] 
cat_cols_low = ['State', 'NewExist', 'UrbanRural'] 

# 1. Label Encoding (Simple)
le = LabelEncoder()
for col in cat_cols_low:
    df_all[col] = df_all[col].astype(str)
    df_all[col] = le.fit_transform(df_all[col])

# Eliminamos originales para que no metan ruido
cols_to_drop = ['id', 'LoanNr_ChkDgt', 'Name', 'ApprovalDate', 'DisbursementDate', 
                'is_train', 'Accept', 'Bank', 'City'] # <--- Borramos Bank y City originales

# Separamos Train y Test
df_train = df_all[df_all['is_train'] == 1].copy()
df_test = df_all[df_all['is_train'] == 0].copy()
y = df_train['Accept'].astype(int)

# 2. Target Encoding (Sobre las columnas agrupadas)
for col in cat_cols_high:
    mapping = df_train.groupby(col)['Accept'].mean()
    
    df_train[f'{col}_TE'] = df_train[col].map(mapping)
    df_test[f'{col}_TE'] = df_test[col].map(mapping)
    
    # Rellenar nulos en test con media global
    global_mean = y.mean()
    df_train[f'{col}_TE'] = df_train[f'{col}_TE'].fillna(global_mean)
    df_test[f'{col}_TE'] = df_test[f'{col}_TE'].fillna(global_mean)
    
    cols_to_drop.append(col) # Borramos la columna _Grouped (texto) y nos quedamos con la _TE (número)

# Crear matrices finales
X = df_train.drop(columns=cols_to_drop)
X_test_final = df_test.drop(columns=cols_to_drop)

# Imputación final
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X_test_imputed = imputer.transform(X_test_final)

# Split de Validación
X_train, X_val, y_train, y_val = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 5. MODELADO Y COMPARACIÓN
# ==========================================
print("\n--- 5. Entrenando y Comparando Modelos ---")

# MODELO A: GEOMÉTRICO (Regresión Logística)
print("   [1/3] Regresión Logística...")
model_geo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
model_geo.fit(X_train, y_train)
pred_geo = model_geo.predict(X_val)
f1_geo = f1_score(y_val, pred_geo, average='macro')
print(f"         >> F1-Score: {f1_geo:.4f}")

# MODELO B: ÁRBOL (Decision Tree)
print("   [2/3] Árbol de Decisión...")
model_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
model_tree.fit(X_train, y_train)
pred_tree = model_tree.predict(X_val)
f1_tree = f1_score(y_val, pred_tree, average='macro')
print(f"         >> F1-Score: {f1_tree:.4f}")

# MODELO C: ENSEMBLE (XGBoost)
print("   [3/3] XGBoost (Ensemble)...")
model_xgb = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    eval_metric='logloss'
)
model_xgb.fit(X_train, y_train)
pred_xgb = model_xgb.predict(X_val)
f1_xgb = f1_score(y_val, pred_xgb, average='macro')
print(f"         >> F1-Score: {f1_xgb:.4f}")

# ==========================================
# 6. SELECCIÓN Y SUBMISSION
# ==========================================
print("\n" + "="*30)
print(f"FINAL: Geo: {f1_geo:.4f} | Tree: {f1_tree:.4f} | XGB: {f1_xgb:.4f}")

best_model = None
model_name = ""

if f1_xgb >= f1_tree and f1_xgb >= f1_geo:
    best_model = model_xgb
    model_name = "XGBoost"
elif f1_tree > f1_geo:
    best_model = model_tree
    model_name = "Decision Tree"
else:
    best_model = model_geo
    model_name = "Logistic Regression"

print(f"¡Ganador: {model_name}! Generando submission...")

if model_name == "Logistic Regression":
    best_model.fit(X_imputed, y)
    final_preds = best_model.predict(X_test_imputed)
else:
    best_model.fit(X_imputed, y)
    final_preds = best_model.predict(X_test_imputed)

submission = pd.DataFrame({'id': test['id'], 'Accept': final_preds})
submission.to_csv('submission_final.csv', index=False)
print("¡Archivo generado!")

# ==========================================
# 7. BONUS: BLENDING (MEZCLA DE MODELOS)
# ==========================================
print("\n--- Generando Ensemble Ponderado (Blending) ---")

# Obtenemos las probabilidades (no solo 0 o 1) de cada modelo para el Test set
# Nota: Para esto, los modelos deben haber sido re-entrenados con X_imputed (como hiciste al final)
# Asegúrate de que best_model es XGBoost, pero re-entrenamos los otros dos rápido para sacar sus probs

model_geo.fit(X_imputed, y)
model_tree.fit(X_imputed, y)
model_xgb.fit(X_imputed, y)

probs_geo = model_geo.predict_proba(X_test_imputed)[:, 1]
probs_tree = model_tree.predict_proba(X_test_imputed)[:, 1]
probs_xgb = model_xgb.predict_proba(X_test_imputed)[:, 1]

# Fórmula Maestra: Damos mucho peso al mejor (XGB) y un poco a los otros para corregir bordes
# Pesos sugeridos: 80% XGB, 15% Tree, 5% Geo
final_probs = (0.80 * probs_xgb) + (0.15 * probs_tree) + (0.05 * probs_geo)

# Convertimos probabilidad a 0 o 1 (Corte en 0.5)
final_preds_blend = (final_probs >= 0.5).astype(int)

# ==========================================
# CÁLCULO DEL F1-SCORE DEL BLEND (Validación)
# ==========================================
print("\n--- Calculando F1-Score del Blend en Validación ---")

# 1. Obtener probabilidades de cada modelo en el set de VALIDACIÓN (X_val)
# Usamos [:, 1] para quedarnos solo con la probabilidad de la clase 1 (Aceptado)
probs_geo_val = model_geo.predict_proba(X_val)[:, 1]
probs_tree_val = model_tree.predict_proba(X_val)[:, 1]
probs_xgb_val = model_xgb.predict_proba(X_val)[:, 1]

# 2. Aplicar la fórmula maestra (mismos pesos que usarás para el final)
# Pesos: 80% XGBoost, 15% Árbol, 5% Regresión
blend_probs_val = (0.80 * probs_xgb_val) + (0.15 * probs_tree_val) + (0.05 * probs_geo_val)

# 3. Convertir probabilidad a predicción final (0 o 1) usando corte de 0.5
blend_preds_val = (blend_probs_val >= 0.5).astype(int)

# 4. Calcular el Score
f1_blend = f1_score(y_val, blend_preds_val, average='macro')

print(f"F1 Individual XGBoost: {f1_xgb:.4f}")
print(f"F1 COLECTIVO (Blend):  {f1_blend:.4f}")

if f1_blend > f1_xgb:
    print("¡Éxito! El blend supera al mejor modelo individual. Úsalo.")
else:
    print("El blend no mejoró el resultado. Mejor envía solo el XGBoost.")

# Guardar
submission_blend = pd.DataFrame({'id': test['id'], 'Accept': final_preds_blend})
submission_blend.to_csv('submission_blend.csv', index=False)
print("¡Archivo 'submission_blend.csv' generado con la fuerza de los 3 modelos!")