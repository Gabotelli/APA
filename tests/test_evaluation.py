"""Check isolation of learned preprocessing and target-encoding cross fitting."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evaluate import LoanFeatures, build_pipeline


class EvaluationIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = pd.read_csv(Path(__file__).resolve().parents[1] / 'data/train.csv',
                              nrows=200, low_memory=False).drop(columns='Accept')
        cls.labels = np.tile([0, 1], 100)

    def test_validation_categories_do_not_change_training_statistics(self):
        features = LoanFeatures('v22').fit(self.rows)
        original_counts = features.bank_counts_.copy()
        validation = self.rows.iloc[:10].copy()
        validation['Bank'] = 'VALIDATION_ONLY_BANK'
        validation['City'] = 'VALIDATION_ONLY_CITY'
        result = features.transform(validation)
        self.assertTrue((result['Bank_Size'] == 0).all())
        self.assertTrue((result['Bank_Grouped'] == 'Other').all())
        self.assertTrue((result['City_Grouped'] == 'Other').all())
        pd.testing.assert_series_equal(original_counts, features.bank_counts_)

    def test_unique_target_categories_are_cross_fitted(self):
        rows = self.rows.copy()
        rows['FranchiseCode'] = np.arange(1000, 1200)
        pipeline = build_pipeline('decision_tree')
        transformed = pipeline[:-1].fit_transform(rows, self.labels)
        # Every franchise occurs once. Its OOF encoding must use a training-fold
        # prior, not that row's label (all internal stratified priors are 0.5).
        np.testing.assert_allclose(transformed[:, -1], 0.5)
        encoded_in_sample = pipeline[:-1].transform(rows)
        self.assertGreater(np.ptp(encoded_in_sample[:, -1]), 0)

    def test_transform_keeps_imputer_and_target_mappings_fixed(self):
        pipeline = build_pipeline('decision_tree').fit(self.rows, self.labels)
        pre = pipeline.named_steps['preprocess']
        medians = pre.named_transformers_['numeric'].statistics_.copy()
        encodings = [x.copy() for x in pre.named_transformers_['target'].encodings_]
        validation = self.rows.iloc[:10].copy()
        validation['DisbursementGross'] = '$999,999,999.00'
        validation['BankState'] = 'UNKNOWN_STATE'
        pipeline.predict_proba(validation)
        np.testing.assert_array_equal(medians, pre.named_transformers_['numeric'].statistics_)
        for old, new in zip(encodings, pre.named_transformers_['target'].encodings_):
            np.testing.assert_array_equal(old, new)


if __name__ == '__main__':
    unittest.main()
