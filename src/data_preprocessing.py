import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


class Preprocessor:
    def __init__(self, drop_id=True, high_missing_thresh=0.60):
        self.drop_id = drop_id
        self.high_missing_thresh = high_missing_thresh
        self.num_imputer = None
        self.cat_imputer = None
        self.scaler = None
        self.low_card_cols = []
        self.high_card_cols = []
        self.freq_maps = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.drop_id:
            drop_cols = [c for c in df.columns if c.lower() in ("id","client_id","application_id")]
            df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        obj_cols = df.select_dtypes(include=['object']).columns.tolist()
        for c in obj_cols:
            s = df[c].astype(str).str.replace(',','').str.replace(' ', '').replace('nan', np.nan)
            coerced = pd.to_numeric(s, errors='coerce')
            if coerced.notnull().sum() / len(coerced) > 0.6:
                df[c] = coerced

        missing_pct = df.isnull().mean()
        drop_high_missing = missing_pct[missing_pct > self.high_missing_thresh].index.tolist()
        df = df.drop(columns=drop_high_missing)

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Default' in num_cols:
            num_cols.remove('Default')
        cat_cols = [c for c in df.columns if c not in num_cols and c != 'Default']

        if num_cols:
            self.num_imputer = SimpleImputer(strategy='median')
            df[num_cols] = self.num_imputer.fit_transform(df[num_cols])
        if cat_cols:
            self.cat_imputer = SimpleImputer(strategy='constant', fill_value='MISSING')
            df[cat_cols] = self.cat_imputer.fit_transform(df[cat_cols])

        if set(['Credit_Amount','Client_Income']).issubset(df.columns):
            df['credit_to_income'] = df['Credit_Amount'] / df['Client_Income'].replace({0: np.nan})
        if 'Loan_Annuity' in df.columns and 'Client_Income' in df.columns:
            df['annuity_to_income'] = df['Loan_Annuity'] / df['Client_Income'].replace({0: np.nan})
        df['credit_to_income'] = df.get('credit_to_income', 0).replace([np.inf, -np.inf], np.nan).fillna(0)
        df['annuity_to_income'] = df.get('annuity_to_income', 0).replace([np.inf, -np.inf], np.nan).fillna(0)

        self.low_card_cols = [c for c in cat_cols if df[c].nunique() <= 10]
        self.high_card_cols = [c for c in cat_cols if df[c].nunique() > 10]

        df = pd.get_dummies(df, columns=self.low_card_cols, drop_first=True)

        for c in self.high_card_cols:
            freq = df[c].value_counts(normalize=True)
            self.freq_maps[c] = freq.to_dict()
            df[c + '_freq_enc'] = df[c].map(self.freq_maps[c]).astype(float)
        df = df.drop(columns=self.high_card_cols)

        numeric_after = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Default' in numeric_after:
            numeric_after.remove('Default')
        self.scaler = StandardScaler()
        df[numeric_after] = self.scaler.fit_transform(df[numeric_after])

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.drop_id:
            drop_cols = [c for c in df.columns if c.lower() in ("id","client_id","application_id")]
            df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        obj_cols = df.select_dtypes(include=['object']).columns.tolist()
        for c in obj_cols:
            s = df[c].astype(str).str.replace(',','').str.replace(' ', '').replace('nan', np.nan)
            coerced = pd.to_numeric(s, errors='coerce')
            if coerced.notnull().sum() / len(coerced) > 0.6:
                df[c] = coerced

        if self.num_imputer is not None:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Default' in num_cols:
                num_cols.remove('Default')
            if num_cols:
                df[num_cols] = self.num_imputer.transform(df[num_cols])
        if self.cat_imputer is not None:
            cat_cols = [c for c in df.columns if c not in df.select_dtypes(include=[np.number]).columns.tolist() and c != 'Default']
            if cat_cols:
                df[cat_cols] = self.cat_imputer.transform(df[cat_cols])

        if set(['Credit_Amount','Client_Income']).issubset(df.columns):
            df['credit_to_income'] = df['Credit_Amount'] / df['Client_Income'].replace({0: np.nan})
        if 'Loan_Annuity' in df.columns and 'Client_Income' in df.columns:
            df['annuity_to_income'] = df['Loan_Annuity'] / df['Client_Income'].replace({0: np.nan})
        df['credit_to_income'] = df.get('credit_to_income', 0).replace([np.inf, -np.inf], np.nan).fillna(0)
        df['annuity_to_income'] = df.get('annuity_to_income', 0).replace([np.inf, -np.inf], np.nan).fillna(0)

        df = pd.get_dummies(df, columns=self.low_card_cols, drop_first=True)

        for c, fmap in self.freq_maps.items():
            df[c + '_freq_enc'] = df[c].map(fmap).astype(float)
        df = df.drop(columns=list(self.freq_maps.keys()), errors='ignore')

        numeric_after = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Default' in numeric_after:
            numeric_after.remove('Default')
        if self.scaler is not None and numeric_after:
            df[numeric_after] = self.scaler.transform(df[numeric_after])

        return df

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)
