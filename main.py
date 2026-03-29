
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier


try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except:
    HAS_SGKF = False


RANDOM_STATE = 42
TARGET = "label_noshow"

train = pd.read_csv("appointments_train.csv")
test = pd.read_csv("appointments_test.csv")
sample_sub = pd.read_csv("sample_submission.csv")

patients = pd.read_csv("patients.csv")
clinics = pd.read_csv("clinics.csv")

print("appointments_train:", train.shape)
print("appointments_test :", test.shape)
print("patients          :", patients.shape)
print("clinics           :", clinics.shape)
print("sample_submission :", sample_sub.shape)
if "specialty" in clinics.columns and "clinic_specialty" not in clinics.columns:
    clinics = clinics.rename(columns={"specialty": "clinic_specialty"})

train = train.merge(patients, on="patient_id", how="left")
test = test.merge(patients, on="patient_id", how="left")

train = train.merge(clinics, on="clinic_id", how="left")
test = test.merge(clinics, on="clinic_id", how="left")

print("\nAfter merge:")
print("train:", train.shape)
print("test :", test.shape)

def add_time_features(df):
    df = df.copy()

 
    if "appointment_datetime" in df.columns:
        df["appointment_datetime"] = pd.to_datetime(
            df["appointment_datetime"], errors="coerce"
        )

    if "booking_datetime" in df.columns:
        df["booking_datetime"] = pd.to_datetime(
            df["booking_datetime"], errors="coerce"
        )
   
    if "appointment_datetime" in df.columns:
        df["appointment_year"] = df["appointment_datetime"].dt.year
        df["appointment_month"] = df["appointment_datetime"].dt.month
        df["appointment_day"] = df["appointment_datetime"].dt.day
        df["appointment_hour"] = df["appointment_datetime"].dt.hour
        df["appointment_minute"] = df["appointment_datetime"].dt.minute
        df["appointment_dow"] = df["appointment_datetime"].dt.weekday
        df["appointment_weekofyear"] = (
            df["appointment_datetime"].dt.isocalendar().week.astype("float")
        )
        df["is_weekend"] = df["appointment_dow"].isin([5, 6]).astype(int)
        df["is_monday"] = (df["appointment_dow"] == 0).astype(int)
        df["is_friday"] = (df["appointment_dow"] == 4).astype(int)
        df["is_morning"] = df["appointment_hour"].between(6, 11).astype(int)
        df["is_lunch_hour"] = df["appointment_hour"].isin([12, 13]).astype(int)
        df["is_evening"] = (df["appointment_hour"] >= 17).astype(int)

        df["hour_group"] = pd.cut(
            df["appointment_hour"],
            bins=[-1, 8, 12, 16, 24],
            labels=["early_morning", "morning", "afternoon", "evening"]
        ).astype("object")

    if "booking_datetime" in df.columns:
        df["booking_year"] = df["booking_datetime"].dt.year
        df["booking_month"] = df["booking_datetime"].dt.month
        df["booking_day"] = df["booking_datetime"].dt.day
        df["booking_hour"] = df["booking_datetime"].dt.hour
        df["booking_dow"] = df["booking_datetime"].dt.weekday
        df["booking_is_weekend"] = df["booking_dow"].isin([5, 6]).astype(int)


    if "appointment_datetime" in df.columns and "booking_datetime" in df.columns:
        diff = df["appointment_datetime"] - df["booking_datetime"]
        df["days_diff"] = diff.dt.days
        df["hours_diff_calc"] = diff.dt.total_seconds() / 3600.0
        df["minutes_diff_calc"] = diff.dt.total_seconds() / 60.0

  
    if "lead_time_hours" in df.columns:
        df["lead_time_days"] = df["lead_time_hours"] / 24.0
        df["lead_time_weeks"] = df["lead_time_hours"] / (24.0 * 7.0)
        df["short_lead"] = (df["lead_time_hours"] <= 24).astype(int)
        df["very_short_lead"] = (df["lead_time_hours"] <= 6).astype(int)
        df["long_lead"] = (df["lead_time_hours"] >= 72).astype(int)

    return df


def add_sms_features(df):
    df = df.copy()

    if "sms_lead_hours" in df.columns:
        df["sms_missing"] = df["sms_lead_hours"].isna().astype(int)
        df["has_sms"] = df["sms_lead_hours"].notna().astype(int)
        df["sms_lead_days"] = df["sms_lead_hours"] / 24.0
        df["late_sms"] = (df["sms_lead_hours"] <= 6).fillna(0).astype(int)
        df["early_sms"] = (df["sms_lead_hours"] >= 24).fillna(0).astype(int)

    return df


def add_ratio_and_interaction_features(df):
    df = df.copy()

    
    if {"lead_time_hours", "wait_mins_est"}.issubset(df.columns):
        df["time_pressure"] = df["lead_time_hours"] / (df["wait_mins_est"] + 1)
        df["wait_x_lead"] = df["wait_mins_est"] * df["lead_time_hours"]

    if {"prior_noshow_rate", "ses_score", "distance_km"}.issubset(df.columns):
        df["behavior_score"] = (
            df["prior_noshow_rate"] * 3
            + df["ses_score"]
            + (df["distance_km"] / 20.0)
        )

    if {"distance_km", "lead_time_hours"}.issubset(df.columns):
        df["distance_x_lead"] = df["distance_km"] * df["lead_time_hours"]
        df["distance_per_lead"] = df["distance_km"] / (df["lead_time_hours"] + 1)

    if {"ses_score", "prior_noshow_rate"}.issubset(df.columns):
        df["ses_x_noshowhist"] = df["ses_score"] * df["prior_noshow_rate"]

    if {"wait_mins_est", "appointment_hour"}.issubset(df.columns):
        df["wait_x_hour"] = df["wait_mins_est"] * df["appointment_hour"]

    if {"lead_time_hours", "appointment_hour"}.issubset(df.columns):
        df["lead_x_hour"] = df["lead_time_hours"] * df["appointment_hour"]

    if {"lead_time_days", "appointment_dow"}.issubset(df.columns):
        df["lead_x_dow"] = df["lead_time_days"] * df["appointment_dow"]

    if {"distance_km", "wait_mins_est"}.issubset(df.columns):
        df["distance_x_wait"] = df["distance_km"] * df["wait_mins_est"]

    return df


def add_geographical_features(df):
    df = df.copy()

    if {"residence_lat", "clinic_lat"}.issubset(df.columns):
        df["lat_diff"] = (df["residence_lat"] - df["clinic_lat"]).abs()

    if {"residence_lon", "clinic_lon"}.issubset(df.columns):
        df["lon_diff"] = (df["residence_lon"] - df["clinic_lon"]).abs()

    if {"lat_diff", "lon_diff"}.issubset(df.columns):
        df["geo_manhattan"] = df["lat_diff"] + df["lon_diff"]

    return df


def add_combination_categoricals(df):
    df = df.copy()

    if {"clinic_id", "appointment_dow"}.issubset(df.columns):
        df["clinic_dow_combo"] = (
            df["clinic_id"].astype(str) + "_" + df["appointment_dow"].astype(str)
        )

    if {"booking_channel", "appointment_hour"}.issubset(df.columns):
        df["channel_hour_combo"] = (
            df["booking_channel"].astype(str) + "_" + df["appointment_hour"].astype(str)
        )

    if {"clinic_specialty", "appointment_dow"}.issubset(df.columns):
        df["specialty_dow_combo"] = (
            df["clinic_specialty"].astype(str) + "_" + df["appointment_dow"].astype(str)
        )

    if {"clinic_specialty", "hour_group"}.issubset(df.columns):
        df["specialty_hour_combo"] = (
            df["clinic_specialty"].astype(str) + "_" + df["hour_group"].astype(str)
        )

    return df


def add_all_features(df):
    df = df.copy()
    df = add_time_features(df)
    df = add_sms_features(df)
    df = add_geographical_features(df)
    df = add_ratio_and_interaction_features(df)
    df = add_combination_categoricals(df)
    return df


def add_frequency_encoding(train_df, test_df, cols):
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in cols:
        if col in train_df.columns and col in test_df.columns:
            freq = train_df[col].astype(str).value_counts(dropna=False)
            train_df[f"{col}_freq"] = train_df[col].astype(str).map(freq).fillna(0)
            test_df[f"{col}_freq"] = test_df[col].astype(str).map(freq).fillna(0)

    return train_df, test_df


def kfold_target_encode(train_df, test_df, col, target, n_splits=5, alpha=20, random_state=42):
    """
    OOF target encoding:
    train tarafı fold içi hesaplanır
    test tarafı full-train mapping ile hesaplanır
    smoothing uygulanır
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    global_mean = train_df[target].mean()
    train_encoded = np.zeros(len(train_df))
    test_encoded = np.zeros(len(test_df))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for tr_idx, val_idx in skf.split(train_df, train_df[target]):
        tr = train_df.iloc[tr_idx]
        val = train_df.iloc[val_idx]

        stats = tr.groupby(col)[target].agg(["mean", "count"])
        smooth = (stats["mean"] * stats["count"] + global_mean * alpha) / (stats["count"] + alpha)

        train_encoded[val_idx] = val[col].map(smooth).fillna(global_mean).values

    stats_full = train_df.groupby(col)[target].agg(["mean", "count"])
    smooth_full = (
        (stats_full["mean"] * stats_full["count"] + global_mean * alpha)
        / (stats_full["count"] + alpha)
    )

    test_encoded = test_df[col].map(smooth_full).fillna(global_mean).values

    return train_encoded, test_encoded
train = add_all_features(train)
test = add_all_features(test)

print("\nFeature engineering tamamlandı.")
print("train:", train.shape)
print("test :", test.shape)

train = train.dropna(subset=[TARGET]).reset_index(drop=True)
test = test.reset_index(drop=True)
freq_cols = [
    "patient_id",
    "clinic_id",
    "clinic_specialty",
    "booking_channel",
    "hour_group",
    "clinic_dow_combo",
    "channel_hour_combo",
    "specialty_dow_combo",
    "specialty_hour_combo",
]

freq_cols = [c for c in freq_cols if c in train.columns and c in test.columns]
train, test = add_frequency_encoding(train, test, freq_cols)

print("\nFrequency encoding columns:")
print(freq_cols)
te_cols = [
    "patient_id",
    "clinic_id",
    "clinic_specialty",
    "booking_channel",
    "hour_group",
    "clinic_dow_combo",
    "channel_hour_combo",
    "specialty_dow_combo",
    "specialty_hour_combo",
]

te_cols = [c for c in te_cols if c in train.columns and c in test.columns]

for col in te_cols:
    train[f"{col}_te"], test[f"{col}_te"] = kfold_target_encode(
        train_df=train,
        test_df=test,
        col=col,
        target=TARGET,
        n_splits=5,
        alpha=20,
        random_state=RANDOM_STATE
    )

print("\nOOF target encoding tamamlandı.")
print("TE columns:", te_cols)
num_cols = train.select_dtypes(include=["int64", "int32", "float64", "float32"]).columns.tolist()
num_cols = [c for c in num_cols if c != TARGET]

for col in num_cols:
    med = train[col].median()
    train[col] = train[col].fillna(med)
    test[col] = test[col].fillna(med)

cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()

for col in cat_cols:
    train[col] = train[col].astype("object").fillna("missing")
    test[col] = test[col].astype("object").fillna("missing")

for col in cat_cols:
    all_cats = pd.Index(train[col].astype(str).unique()).union(
        pd.Index(test[col].astype(str).unique())
    )
    train[col] = pd.Categorical(train[col].astype(str), categories=all_cats)
    test[col] = pd.Categorical(test[col].astype(str), categories=all_cats)

print("\nMissing value handling tamamlandı.")
print("Categorical columns:", cat_cols)
drop_cols = [
    "appointment_id",
    TARGET,
    "appointment_datetime",
    "booking_datetime",
]


drop_cols = [c for c in drop_cols if c in train.columns]

features = [c for c in train.columns if c not in drop_cols]


X = train[features].copy()
y = train[TARGET].copy()
X_test = test[features].copy()

print("\nFeature count:", len(features))
use_group_cv = HAS_SGKF and ("patient_id" in train.columns)

if use_group_cv:
    print("\nStratifiedGroupKFold kullanılacak (group = patient_id).")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(X, y, groups=train["patient_id"]))
else:
    print("\nStratifiedKFold kullanılacak.")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(X, y))

from xgboost import XGBClassifier 
cat_feature_indices = [X.columns.get_loc(col) for col in cat_cols if col in X.columns]

def make_xgb_compatible(df):
    df = df.copy()
    for col in df.columns:
        if str(df[col].dtype) in ["category", "object"]:
            df[col] = pd.Categorical(df[col]).codes
    return df


lgb_params = dict(
    n_estimators=1000,
    learning_rate=0.02,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=40,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.80,
    reg_alpha=0.5,
    reg_lambda=1.5,
    objective="binary",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1
)

xgb_params = dict(
    n_estimators=1000,
    learning_rate=0.025,
    max_depth=6,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.80,
    reg_alpha=0.3,
    reg_lambda=1.5,
    gamma=0.1,
    objective="binary:logistic",
    eval_metric="aucpr",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist",
    enable_categorical=False
)

cat_params = dict(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=5.0,
    loss_function="Logloss",
    eval_metric="PRAUC",
    random_seed=RANDOM_STATE,
    verbose=False
)


lgb_oof = np.zeros(len(train))
xgb_oof = np.zeros(len(train))
cat_oof = np.zeros(len(train))

lgb_test_preds = np.zeros(len(test))
xgb_test_preds = np.zeros(len(test))
cat_test_preds = np.zeros(len(test))

lgb_scores = []
xgb_scores = []
cat_scores = []
ens_scores = []

feature_importance_list = []

for fold, (tr_idx, val_idx) in enumerate(splits, 1):
    print(f"\n{'='*20} Fold {fold} {'='*20}")

    X_train = X.iloc[tr_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_train = y.iloc[tr_idx].copy()
    y_val = y.iloc[val_idx].copy()

    X_test_fold = X_test.copy()

   
    lgb_model = LGBMClassifier(**lgb_params)

    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        categorical_feature=[c for c in cat_cols if c in X.columns],
        callbacks=[
            early_stopping(stopping_rounds=150, verbose=False),
            log_evaluation(period=0)
        ]
    )

    lgb_val_pred = lgb_model.predict_proba(X_val)[:, 1]
    lgb_test_pred = lgb_model.predict_proba(X_test_fold)[:, 1]

    lgb_oof[val_idx] = lgb_val_pred
    lgb_test_preds += lgb_test_pred / len(splits)

    lgb_fold_score = average_precision_score(y_val, lgb_val_pred)
    lgb_scores.append(lgb_fold_score)

    fold_importance = pd.DataFrame({
        "feature": features,
        "importance": lgb_model.feature_importances_,
        "fold": fold
    })
    feature_importance_list.append(fold_importance)

   
    X_train_xgb = make_xgb_compatible(X_train)
    X_val_xgb = make_xgb_compatible(X_val)
    X_test_xgb = make_xgb_compatible(X_test_fold)

    xgb_model = XGBClassifier(**xgb_params)

    xgb_model.fit(
        X_train_xgb,
        y_train,
        eval_set=[(X_val_xgb, y_val)],
        verbose=False
    )

    xgb_val_pred = xgb_model.predict_proba(X_val_xgb)[:, 1]
    xgb_test_pred = xgb_model.predict_proba(X_test_xgb)[:, 1]

    xgb_oof[val_idx] = xgb_val_pred
    xgb_test_preds += xgb_test_pred / len(splits)

    xgb_fold_score = average_precision_score(y_val, xgb_val_pred)
    xgb_scores.append(xgb_fold_score)

  
     
    cat_model = CatBoostClassifier(**cat_params)

    cat_model.fit(
         X_train,
         y_train,
         eval_set=(X_val, y_val),
         cat_features=cat_feature_indices,
         use_best_model=True
     )

    cat_val_pred = cat_model.predict_proba(X_val)[:, 1]
    cat_test_pred = cat_model.predict_proba(X_test_fold)[:, 1]

    cat_oof[val_idx] = cat_val_pred
    cat_test_preds += cat_test_pred / len(splits)

    cat_fold_score = average_precision_score(y_val, cat_val_pred)
    cat_scores.append(cat_fold_score)

    ens_val_pred = 0.40 * lgb_val_pred + 0.30 * xgb_val_pred + 0.30 * cat_val_pred
    ens_fold_score = average_precision_score(y_val, ens_val_pred)
    ens_scores.append(ens_fold_score)

    print(f"LightGBM Fold {fold} PR-AUC : {lgb_fold_score:.6f}")
    print(f"XGBoost  Fold {fold} PR-AUC : {xgb_fold_score:.6f}")
    print(f"CatBoost Fold {fold} PR-AUC : {cat_fold_score:.6f}")
    print(f"Ensemble Fold {fold} PR-AUC : {ens_fold_score:.6f}")


lgb_oof_score = average_precision_score(y, lgb_oof)
xgb_oof_score = average_precision_score(y, xgb_oof)
cat_oof_score = average_precision_score(y, cat_oof)

ensemble_oof = 0.40 * lgb_oof + 0.30 * xgb_oof + 0.30 * cat_oof
ensemble_oof_score = average_precision_score(y, ensemble_oof)

print("\n" + "="*50)
print("MODEL BAZLI SONUÇLAR")
print("="*50)

print("\nLightGBM CV scores:", lgb_scores)
print("LightGBM Mean CV PR-AUC:", np.mean(lgb_scores))
print("LightGBM OOF PR-AUC    :", lgb_oof_score)

print("\nXGBoost CV scores:", xgb_scores)
print("XGBoost Mean CV PR-AUC:", np.mean(xgb_scores))
print("XGBoost OOF PR-AUC    :", xgb_oof_score)

print("\nCatBoost CV scores:", cat_scores)
print("CatBoost Mean CV PR-AUC:", np.mean(cat_scores))
print("CatBoost OOF PR-AUC    :", cat_oof_score)

print("\nEnsemble CV scores:", ens_scores)
print("Ensemble Mean CV PR-AUC:", np.mean(ens_scores))
print("Ensemble OOF PR-AUC    :", ensemble_oof_score)
feature_importance = pd.concat(feature_importance_list, axis=0)
feature_importance = (
    feature_importance.groupby("feature")["importance"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

print("\nTop 40 LightGBM feature importance:")
print(feature_importance.head(40))

submission = sample_sub.copy()

if "appointment_id" in test.columns:
    submission["appointment_id"] = test["appointment_id"]


sub_lgb = submission.copy()
sub_lgb[TARGET] = lgb_test_preds
sub_lgb.to_csv("submission_lgbm_enhanced.csv", index=False)


sub_xgb = submission.copy()
sub_xgb[TARGET] = xgb_test_preds
sub_xgb.to_csv("submission_xgb_enhanced.csv", index=False)


sub_cat = submission.copy()
sub_cat[TARGET] = cat_test_preds
sub_cat.to_csv("submission_catboost_enhanced.csv", index=False)


ensemble_test_preds = 0.40 * lgb_test_preds + 0.30 * xgb_test_preds + 0.30 * cat_test_preds
sub_ens = submission.copy()
sub_ens[TARGET] = ensemble_test_preds
sub_ens.to_csv("submission_ensemble_enhanced.csv", index=False)

print("\nOluşturulan dosyalar:")
print("- submission_lgbm_enhanced.csv")
print("- submission_xgb_enhanced.csv")
print("- submission_catboost_enhanced.csv")
print("- submission_ensemble_enhanced.csv")
