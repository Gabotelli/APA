"""Inductive evaluation added after the competition; historical scripts are separate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, TargetEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = [
    'ApprovalFY', 'NoEmp', 'CreateJob', 'RetainedJob', 'RevLineCr', 'LowDoc',
    'DisbursementGross', 'ApprovalDate_Year', 'DisbursementDate_Year',
    'Days_To_Disbursement', 'Legal_Entity', 'Sector_Risk', 'StateSame',
    'IsFranchise', 'Loan_Per_Emp', 'TotalJobs', 'RevLine_Urban',
]


class LoanFeatures(TransformerMixin, BaseEstimator):
    """Row-wise competition features; learned frequencies use training rows only."""

    def __init__(self, variant='v21'):
        self.variant = variant

    def fit(self, X, y=None):
        self.banks_ = set(X['Bank'].value_counts().head(50).index)
        self.cities_ = set(X['City'].value_counts().head(80).index)
        self.bank_counts_ = X['Bank'].value_counts()
        return self

    def transform(self, X):
        d = X.copy()
        d['DisbursementGross'] = pd.to_numeric(
            d['DisbursementGross'].astype(str).str.replace(r'[$,]', '', regex=True),
            errors='coerce',
        )
        d['ApprovalFY'] = pd.to_numeric(
            d['ApprovalFY'].astype(str).str.replace('A', '', regex=False), errors='coerce'
        )
        for col in ['ApprovalDate', 'DisbursementDate']:
            dates = pd.to_datetime(d[col], errors='coerce', format='mixed')
            d[col] = dates
            d[col + '_Year'] = dates.dt.year.where(dates.dt.year <= 2025, dates.dt.year - 100)
        d['Days_To_Disbursement'] = (d['DisbursementDate'] - d['ApprovalDate']).dt.days
        name = d['Name'].fillna('').astype(str).str.upper()
        d['Legal_Entity'] = name.str.contains('LLC|INC|CORP|LTD', regex=True).astype(int)
        risk = 'REALTY|ESTATE|CONST|BUILD' + ('|DEV' if self.variant == 'v21' else '')
        d['Sector_Risk'] = name.str.contains(risk, regex=True).astype(int)
        for col in ['RevLineCr', 'LowDoc']:
            d[col] = d[col].astype(str).str.upper().isin(['Y', 'T', '1']).astype(int)
        d['StateSame'] = (d['State'] == d['BankState']).astype(int)
        d['IsFranchise'] = (d['FranchiseCode'] > 1).astype(int)
        d['NewExist'] = d['NewExist'].fillna(1).replace(0, 1)
        d['NoEmp'] = d['NoEmp'].replace(0, 1)
        d['Loan_Per_Emp'] = d['DisbursementGross'] / d['NoEmp']
        d['TotalJobs'] = d['CreateJob'] + d['RetainedJob']
        d['RevLine_Urban'] = d['RevLineCr'] * d['UrbanRural']
        d['Bank_Grouped'] = d['Bank'].where(d['Bank'].isin(self.banks_), 'Other')
        d['City_Grouped'] = d['City'].where(d['City'].isin(self.cities_), 'Other')
        ordinal = ['NewExist', 'UrbanRural', 'BankState', 'State']
        numeric = list(NUMERIC)
        if self.variant == 'v22':
            d['Bank_Size'] = d['Bank'].map(self.bank_counts_).fillna(0)
            d['CompanySize_Cat'] = pd.cut(
                d['NoEmp'], [-1, 1, 5, 20, 99999], labels=['Micro', 'Small', 'Medium', 'Large']
            )
            numeric.append('Bank_Size')
            ordinal.append('CompanySize_Cat')
        target = ['Bank_Grouped', 'City_Grouped', 'FranchiseCode']
        for col in ordinal + target:
            d[col] = d[col].astype('string').fillna('__missing__').astype(str)
        return d[numeric + ordinal + target].replace([np.inf, -np.inf], np.nan)


def build_pipeline(model_name, quick=False, seed=42, jobs=1):
    variant = 'v22' if model_name == 'v22' else 'v21'
    numeric = NUMERIC + (['Bank_Size'] if variant == 'v22' else [])
    ordinal = ['NewExist', 'UrbanRural', 'BankState', 'State']
    if variant == 'v22':
        ordinal.append('CompanySize_Cat')
    preprocess = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median', keep_empty_features=True), numeric),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal),
        ('target', TargetEncoder(target_type='binary', smooth=20, cv=5,
                                 shuffle=True, random_state=seed),
         ['Bank_Grouped', 'City_Grouped', 'FranchiseCode']),
    ])
    if model_name == 'decision_tree':
        model = DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=seed)
    elif model_name == 'logistic_regression':
        model = Pipeline([
            ('scale', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed)),
        ])
    else:
        import xgboost as xgb
        n_xgb, n_rf, n_hist = (20, 20, 20) if quick else (1000, 500, 500)
        constraints = tuple(1 if c == 'DisbursementGross' else 0
                            for c in numeric + ordinal + ['Bank_TE', 'City_TE', 'Franchise_TE'])
        xgb_model = xgb.XGBClassifier(
            n_estimators=n_xgb, learning_rate=0.015, max_depth=8,
            subsample=0.7, colsample_bytree=0.7, n_jobs=jobs, random_state=seed,
            **({'monotone_constraints': constraints} if variant == 'v22' else {}),
        )
        rf = RandomForestClassifier(n_estimators=n_rf, max_depth=15,
                                    class_weight='balanced', n_jobs=jobs, random_state=seed)
        estimators = [('xgb', xgb_model), ('rf', rf)]
        weights = [6, 4]
        if variant == 'v21':
            estimators.append(('hgb', HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=n_hist, max_depth=10,
                l2_regularization=0.1, random_state=seed,
            )))
            weights = [5, 3, 2]
        model = BalancedVotingClassifier(estimators=estimators, voting='soft', weights=weights)
    return Pipeline([('features', LoanFeatures(variant)), ('preprocess', preprocess), ('model', model)])


class BalancedVotingClassifier(VotingClassifier):
    """Recompute balanced weights inside each fit, including each outer CV fold."""

    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is None:
            sample_weight = compute_sample_weight('balanced', y)
        return super().fit(X, y, sample_weight=sample_weight, **fit_params)


def predictions(probabilities, model_name):
    if model_name == 'v22':
        return (probabilities[:, 1] >= 0.45).astype(int)
    return probabilities.argmax(axis=1)


def evaluate(args):
    raw = pd.read_csv(args.data, low_memory=False)
    if not raw['Accept'].isin([0, 1]).all():
        raise ValueError('Accept must contain only 0 and 1.')
    X, y = raw.drop(columns='Accept'), raw['Accept'].astype(int)
    if args.max_rows and args.max_rows < len(raw):
        X, _, y, _ = train_test_split(X, y, train_size=args.max_rows, stratify=y,
                                    random_state=args.seed)
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    report = {
        'seed': args.seed, 'split': 'stratified random rows; 80% development / 20% holdout',
        'development_rows': len(X_train), 'holdout_rows': len(X_holdout),
        'outer_cv_folds': args.folds, 'target_encoding_folds': 5,
        'quick_check': args.quick, 'sampled_rows': args.max_rows or None,
        'pseudo_labeling': False, 'models': {},
        'limitations': [
            'Random row splitting does not establish temporal or entity-independent performance.',
            'Disbursement feature availability at the prediction date is not verified.',
            'This inductive evaluation does not reproduce the transductive competition submissions.',
            'Repeated holdout inspection during model selection would invalidate its independence.',
        ],
    }
    for name in args.models:
        pipeline = build_pipeline(name, args.quick, args.seed, args.jobs)
        oof_probs = cross_val_predict(pipeline, X_train, y_train, cv=cv,
                                     method='predict_proba', n_jobs=1)
        pipeline.fit(X_train, y_train)
        holdout_preds = predictions(pipeline.predict_proba(X_holdout), name)
        report['models'][name] = {
            'oof_macro_f1': float(f1_score(y_train, predictions(oof_probs, name), average='macro')),
            'holdout_macro_f1': float(f1_score(y_holdout, holdout_preds, average='macro')),
            'holdout_confusion_matrix': confusion_matrix(y_holdout, holdout_preds, labels=[0, 1]).tolist(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/train.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/evaluation/report.json')
    parser.add_argument('--models', nargs='+', choices=['decision_tree', 'logistic_regression', 'v21', 'v22'],
                        default=['decision_tree', 'logistic_regression', 'v21', 'v22'])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--quick', action='store_true', help='20 estimators/iterations; smoke check only')
    parser.add_argument('--max-rows', type=int, default=0, help='Optional stratified sample for a smoke check')
    args = parser.parse_args()
    if args.folds < 2 or args.jobs < 1 or args.max_rows < 0:
        parser.error('Require folds >= 2, jobs >= 1 and max-rows >= 0.')
    evaluate(args)


if __name__ == '__main__':
    main()
