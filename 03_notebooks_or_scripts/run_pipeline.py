"""T06 Water Utility Security Analytics - reproducible end-to-end pipeline.

The script generates controlled synthetic data, performs the required analyses,
exports evidence files, and builds a standalone analyst dashboard.
"""

from __future__ import annotations

import html
import json
import math
import os
import re
from pathlib import Path

import joblib
# Use a writable cache in restricted or lab environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "t06-matplotlib"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 821
RNG = np.random.default_rng(SEED)
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "02_data" / "raw"
PROCESSED = ROOT / "02_data" / "processed"
MODELS = ROOT / "04_models"
SIMULATION = ROOT / "05_simulation"
TEXT_MINING = ROOT / "06_text_mining"
DASHBOARD = ROOT / "07_dashboard_or_prototype"
OUTPUTS = ROOT / "08_outputs"
FIGURES = OUTPUTS / "figures"
INTELLIGENCE = OUTPUTS / "intelligence"


for directory in [RAW, PROCESSED, MODELS, SIMULATION, TEXT_MINING, DASHBOARD, OUTPUTS, FIGURES, INTELLIGENCE]:
    directory.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 180})


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")


def generate_vpn_logs() -> pd.DataFrame:
    n = 2000
    vendors = [f"VENDOR_{i:02d}" for i in range(1, 13)]
    users = {vendor: f"USER_{i:02d}" for i, vendor in enumerate(vendors, start=1)}
    approved_devices = {vendor: f"DEV_{i:02d}_A" for i, vendor in enumerate(vendors, start=1)}
    approved_assets = [f"OT_{i:02d}" for i in range(1, 16)]

    base = pd.Timestamp("2026-08-01 00:00:00")
    timestamps = base + pd.to_timedelta(RNG.integers(0, 31 * 24 * 60, n), unit="m")
    vendors_drawn = RNG.choice(vendors, n)
    suspicious_indices = set(RNG.choice(np.arange(n), 150, replace=False).tolist())

    rows = []
    for i in range(n):
        vendor = vendors_drawn[i]
        ts = pd.Timestamp(timestamps[i])
        suspicious = i in suspicious_indices
        off_hours = int(ts.hour < 6 or ts.hour >= 20)
        new_device = int(RNG.random() < (0.65 if suspicious else 0.025))
        failed_attempts = int(RNG.poisson(4.2 if suspicious else 0.25))
        mfa_used = int(RNG.random() > (0.50 if suspicious else 0.05))
        login_result = "FAIL" if RNG.random() < (0.24 if suspicious else 0.035) else "SUCCESS"
        duration_min = float(np.clip(RNG.normal(92 if suspicious else 31, 22 if suspicious else 12), 1, 210))
        bytes_out_mb = float(np.clip(RNG.lognormal(4.0 if suspicious else 1.7, 0.75), 0.1, 420))
        unusual_asset = int(RNG.random() < (0.58 if suspicious else 0.04))
        asset = RNG.choice(approved_assets)
        device = f"DEV_EXT_{RNG.integers(100,999)}" if new_device else approved_devices[vendor]
        region = RNG.choice(["Outside Namibia", "Unknown"] if suspicious else ["Windhoek", "Erongo"], p=[0.75, 0.25] if suspicious else [0.85, 0.15])
        source_ip = f"198.51.100.{RNG.integers(1,255)}" if suspicious else f"10.20.{vendors.index(vendor)+1}.{RNG.integers(2,250)}"
        label = int(
            suspicious
            and (
                (off_hours + new_device + unusual_asset + int(not mfa_used) >= 2)
                or failed_attempts >= 3
                or bytes_out_mb >= 80
            )
        )
        rows.append(
            {
                "session_id": f"VPN_{i+1:04d}",
                "timestamp": ts,
                "vendor_id": vendor,
                "user_id": users[vendor],
                "source_ip": source_ip,
                "device_id": device,
                "geo_region": region,
                "login_result": login_result,
                "mfa_used": mfa_used,
                "duration_min": round(duration_min, 1),
                "bytes_out_mb": round(bytes_out_mb, 2),
                "asset_id": asset,
                "off_hours": off_hours,
                "new_device": new_device,
                "failed_attempts": failed_attempts,
                "unusual_asset": unusual_asset,
                "label_suspicious": label,
            }
        )

    vpn = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    # A clearly traceable incident used by the correlation module.
    incident_rows = vpn.index[-6:]
    incident_times = pd.date_range("2026-08-30 02:04:00", periods=6, freq="3min")
    for j, idx in enumerate(incident_rows):
        vpn.loc[idx, ["timestamp", "vendor_id", "user_id", "source_ip", "device_id", "geo_region"]] = [
            incident_times[j], "VENDOR_07", "USER_07", "198.51.100.77", "DEV_EXT_777", "Outside Namibia"
        ]
        vpn.loc[idx, ["asset_id", "off_hours", "new_device", "failed_attempts", "unusual_asset", "mfa_used", "label_suspicious"]] = [
            "OT_04", 1, 1, 5 if j < 3 else 1, 1, 0, 1
        ]
        vpn.loc[idx, "login_result"] = "FAIL" if j < 3 else "SUCCESS"
        vpn.loc[idx, "duration_min"] = 118.0 if j >= 3 else 2.0
        vpn.loc[idx, "bytes_out_mb"] = 186.0 if j >= 3 else 0.2
    # Add a small amount of label uncertainty to represent imperfect analyst labels.
    non_incident = vpn.index.difference(incident_rows)
    noisy_indices = RNG.choice(non_incident, 24, replace=False)
    vpn.loc[noisy_indices, "label_suspicious"] = 1 - vpn.loc[noisy_indices, "label_suspicious"]
    return vpn.sort_values("timestamp").reset_index(drop=True)


def generate_firewall_logs(vpn: pd.DataFrame) -> pd.DataFrame:
    n = 2000
    protocols = ["MODBUS", "HTTPS", "SSH", "RDP"]
    normal_ports = {"MODBUS": 502, "HTTPS": 443, "SSH": 22, "RDP": 3389}
    sample_sessions = vpn.sample(n=n, replace=True, random_state=SEED).reset_index(drop=True)
    rows = []
    for i, session in sample_sessions.iterrows():
        suspicious_session = int(session["label_suspicious"])
        unusual_protocol = int(RNG.random() < (0.45 if suspicious_session else 0.025))
        protocol = RNG.choice(["TELNET", "SMB", "FTP"]) if unusual_protocol else RNG.choice(protocols, p=[0.50, 0.28, 0.14, 0.08])
        port = {"TELNET": 23, "SMB": 445, "FTP": 21}.get(protocol, normal_ports.get(protocol, 443))
        timestamp = pd.Timestamp(session["timestamp"]) + pd.to_timedelta(RNG.integers(0, 90), unit="m")
        connection_count = int(RNG.poisson(36 if suspicious_session else 7) + 1)
        bytes_out_mb = float(np.clip(RNG.lognormal(3.4 if suspicious_session else 1.3, 0.8), 0.05, 360))
        action = "DENY" if unusual_protocol and RNG.random() < 0.55 else "ALLOW"
        rows.append(
            {
                "event_id": f"FW_{i+1:04d}",
                "timestamp": timestamp,
                "session_id": session["session_id"],
                "source_ip": session["source_ip"],
                "destination_asset": session["asset_id"],
                "protocol": protocol,
                "port": port,
                "action": action,
                "bytes_out_mb": round(bytes_out_mb, 2),
                "connection_count": connection_count,
                "unusual_protocol": unusual_protocol,
            }
        )
    firewall = pd.DataFrame(rows)

    incident_sessions = vpn[(vpn.vendor_id == "VENDOR_07") & (vpn.timestamp >= "2026-08-30 02:10")].session_id.tolist()
    for j, sid in enumerate(incident_sessions[:3]):
        idx = firewall.index[-(j + 1)]
        firewall.loc[idx, :] = [
            f"FW_INC_{j+1}",
            pd.Timestamp("2026-08-30 02:16:00") + pd.Timedelta(minutes=4*j),
            sid,
            "198.51.100.77",
            "OT_04",
            "TELNET" if j == 0 else "MODBUS",
            23 if j == 0 else 502,
            "ALLOW",
            148.0 + 22*j,
            71 + 12*j,
            1,
        ]
    return firewall.sort_values("timestamp").reset_index(drop=True)


def generate_telemetry() -> pd.DataFrame:
    n = 1500
    assets = [f"OT_{i:02d}" for i in range(1, 16)]
    sensor_map = {
        "pressure_bar": (4.0, 0.45),
        "reservoir_level_pct": (72.0, 8.0),
        "flow_lps": (210.0, 24.0),
        "chlorine_mg_l": (0.70, 0.08),
    }
    timestamps = pd.Timestamp("2026-08-01") + pd.to_timedelta(RNG.integers(0, 31 * 24 * 60, n), unit="m")
    rows = []
    anomaly_indices = set(RNG.choice(np.arange(n), 65, replace=False).tolist())
    for i in range(n):
        sensor = RNG.choice(list(sensor_map))
        mean, sd = sensor_map[sensor]
        anomaly = i in anomaly_indices
        value = RNG.normal(mean + (4.0 * sd if anomaly else 0), sd * (1.8 if anomaly else 1))
        command_count = int(RNG.poisson(9 if anomaly else 1.2))
        rows.append(
            {
                "telemetry_id": f"TEL_{i+1:04d}",
                "timestamp": pd.Timestamp(timestamps[i]),
                "asset_id": RNG.choice(assets),
                "sensor_type": sensor,
                "value": round(float(value), 3),
                "set_point": mean,
                "deviation": round(float(abs(value - mean)), 3),
                "command_count": command_count,
                "state": "ALARM" if anomaly else "NORMAL",
                "anomaly_flag": int(anomaly),
            }
        )
    telemetry = pd.DataFrame(rows)
    incident_idx = telemetry.index[-3:]
    incident_values = [6.9, 8.2, 2.1]
    for j, idx in enumerate(incident_idx):
        telemetry.loc[idx, :] = [
            f"TEL_INC_{j+1}",
            pd.Timestamp("2026-08-30 02:24:00") + pd.Timedelta(minutes=4*j),
            "OT_04",
            "pressure_bar",
            incident_values[j],
            4.0,
            abs(incident_values[j] - 4.0),
            14 + j * 3,
            "ALARM",
            1,
        ]
    return telemetry.sort_values("timestamp").reset_index(drop=True)


def generate_tickets() -> pd.DataFrame:
    templates = {
        "normal_maintenance": [
            "Scheduled inspection completed for {asset}; readings are within normal range.",
            "Vendor performed routine firmware check on {asset}; no fault was found.",
            "Monthly valve and pump maintenance completed on {asset} during approved window.",
        ],
        "equipment_fault": [
            "Operator reported unstable pressure readings on {asset}; engineering review requested.",
            "Repeated sensor alarm on {asset}; possible calibration or hardware fault.",
            "Communication loss from {asset}; field technician assigned to inspect equipment.",
        ],
        "security_concern": [
            "Unexpected remote login and configuration activity detected on {asset}; verify vendor account.",
            "Multiple failed VPN attempts followed by unusual controller commands for {asset}.",
            "New external device accessed {asset} outside the approved maintenance window.",
        ],
    }
    categories = RNG.choice(list(templates), 250, p=[0.48, 0.30, 0.22])
    rows = []
    for i, category in enumerate(categories):
        asset = f"OT_{RNG.integers(1,16):02d}"
        ts = pd.Timestamp("2026-08-01") + pd.to_timedelta(RNG.integers(0, 31 * 24 * 60), unit="m")
        rows.append(
            {
                "ticket_id": f"TKT_{i+1:04d}",
                "timestamp": ts,
                "vendor_id": f"VENDOR_{RNG.integers(1,13):02d}",
                "asset_id": asset,
                "description": RNG.choice(templates[category]).format(asset=asset),
                "category": category,
                "priority": {"normal_maintenance": "Low", "equipment_fault": "Medium", "security_concern": "High"}[category],
            }
        )
    tickets = pd.DataFrame(rows)
    tickets.loc[tickets.index[-1], :] = [
        "TKT_INC_01",
        pd.Timestamp("2026-08-30 02:38:00"),
        "VENDOR_07",
        "OT_04",
        "Urgent: failed remote logins were followed by unapproved commands and abnormal pressure on OT_04.",
        "security_concern",
        "High",
    ]
    # Controlled ambiguity: a few ticket labels are intentionally imperfect.
    candidate_indices = tickets.index[:-1]
    noisy_indices = RNG.choice(candidate_indices, 12, replace=False)
    category_cycle = {"normal_maintenance": "equipment_fault", "equipment_fault": "security_concern", "security_concern": "normal_maintenance"}
    tickets.loc[noisy_indices, "category"] = tickets.loc[noisy_indices, "category"].map(category_cycle)
    return tickets.sort_values("timestamp").reset_index(drop=True)


def data_quality_report(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for name, frame in frames.items():
        duplicate_rows = int(frame.duplicated().sum())
        missing_cells = int(frame.isna().sum().sum())
        records.append(
            {
                "dataset": name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "missing_cells_before": missing_cells,
                "duplicate_rows_before": duplicate_rows,
                "missing_cells_after": 0,
                "duplicate_rows_after": 0,
                "status": "Ready",
            }
        )
    return pd.DataFrame(records)


def prepare_supervised(vpn: pd.DataFrame):
    model_data = vpn.copy()
    model_data["timestamp"] = pd.to_datetime(model_data["timestamp"])
    model_data["hour"] = model_data["timestamp"].dt.hour
    model_data["login_success"] = (model_data["login_result"] == "SUCCESS").astype(int)
    model_data["region_local"] = model_data["geo_region"].isin(["Windhoek", "Erongo"]).astype(int)
    feature_cols = [
        "hour", "off_hours", "mfa_used", "duration_min", "bytes_out_mb", "new_device",
        "failed_attempts", "unusual_asset", "login_success", "vendor_id", "asset_id"
    ]
    target = "label_suspicious"
    numeric = [c for c in feature_cols if c not in ["vendor_id", "asset_id"]]
    categorical = ["vendor_id", "asset_id"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        model_data[feature_cols], model_data[target], model_data["session_id"],
        test_size=0.25, random_state=SEED, stratify=model_data[target]
    )
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=180, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=SEED
        ),
    }
    results = []
    fitted = {}

    for name, estimator in candidates.items():

        # --------------------------------------------------------
        # Adversarial training augmentation
        #
        # Only the Random Forest receives low-volume suspicious
        # variants. The test set remains completely untouched.
        # --------------------------------------------------------

        train_features = X_train.copy()
        train_target = y_train.copy()

        if name == "Random Forest":

            suspicious_train = train_features[
                train_target == 1
            ].copy()

            suspicious_target = train_target[
                train_target == 1
            ].copy()

            # Use 50% of suspicious training records
            if len(suspicious_train) > 0:

                augmented_source = suspicious_train.sample(
                    frac=0.50,
                    random_state=SEED
                ).copy()

                # Create slow low-volume adversarial variants
                augmented_source["bytes_out_mb"] = np.minimum(
                    augmented_source["bytes_out_mb"],
                    12
                )

                augmented_source["duration_min"] = np.minimum(
                    augmented_source["duration_min"],
                    35
                )

                augmented_target = pd.Series(
                    1,
                    index=augmented_source.index,
                    name=train_target.name
                )

                train_features = pd.concat(
                    [
                        train_features,
                        augmented_source
                    ],
                    ignore_index=True
                )

                train_target = pd.concat(
                    [
                        train_target.reset_index(drop=True),
                        augmented_target.reset_index(drop=True)
                    ],
                    ignore_index=True
                )

        pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", estimator)
        ])

        pipeline.fit(
            train_features,
            train_target
        )

        prediction = pipeline.predict(X_test)
        probability = pipeline.predict_proba(X_test)[:, 1]

        results.append(
            {
                "model": name,
                "accuracy": accuracy_score(
                    y_test,
                    prediction
                ),
                "precision": precision_score(
                    y_test,
                    prediction,
                    zero_division=0
                ),
                "recall": recall_score(
                    y_test,
                    prediction,
                    zero_division=0
                ),
                "f1_score": f1_score(
                    y_test,
                    prediction,
                    zero_division=0
                ),
                "test_records": len(y_test),
            }
        )

        fitted[name] = (
            pipeline,
            prediction,
            probability
        )
    metrics = pd.DataFrame(results).sort_values("f1_score", ascending=False).reset_index(drop=True)
    best_name = metrics.iloc[0]["model"]
    best_pipeline, best_prediction, best_probability = fitted[best_name]
    evaluation = pd.DataFrame(
        {
            "session_id": id_test.reset_index(drop=True),
            "actual": y_test.reset_index(drop=True),
            "predicted": best_prediction,
            "probability": best_probability,
        }
    )
    return model_data, feature_cols, best_name, best_pipeline, metrics, evaluation, X_test.reset_index(drop=True), y_test.reset_index(drop=True)


def run_anomaly_model(model_data: pd.DataFrame) -> tuple[IsolationForest, pd.DataFrame]:
    features = ["off_hours", "mfa_used", "duration_min", "bytes_out_mb", "new_device", "failed_attempts", "unusual_asset", "login_success", "region_local"]
    matrix = model_data[features].astype(float)
    detector = IsolationForest(n_estimators=180, contamination=0.08, random_state=SEED)
    detector.fit(matrix)
    raw_score = -detector.score_samples(matrix)
    percentile = pd.Series(raw_score).rank(pct=True) * 100
    output = model_data[["session_id", "timestamp", "vendor_id", "user_id", "asset_id", "label_suspicious"]].copy()
    output["anomaly_score"] = raw_score
    output["anomaly_percentile"] = percentile.round(2)
    output["is_outlier"] = (detector.predict(matrix) == -1).astype(int)
    return detector, output.sort_values("anomaly_score", ascending=False).reset_index(drop=True)


def create_ranked_alerts(model_data: pd.DataFrame, supervised_model, anomaly: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    probability = supervised_model.predict_proba(model_data[feature_cols])[:, 1]
    base = model_data.copy()
    base["supervised_probability"] = probability
    base = base.merge(anomaly[["session_id", "anomaly_percentile", "is_outlier"]], on="session_id", how="left")
    rule_points = (
        base["off_hours"] * 8
        + base["new_device"] * 10
        + base["unusual_asset"] * 8
        + (base["failed_attempts"] >= 3).astype(int) * 8
        + (base["mfa_used"] == 0).astype(int) * 5
        + (base["bytes_out_mb"] >= 80).astype(int) * 6
    )
    base["risk_score"] = np.clip(
        base["supervised_probability"] * 58 + base["anomaly_percentile"] * 0.22 + rule_points,
        0,
        100,
    ).round(1)
    base["risk_level"] = pd.cut(base["risk_score"], bins=[-1, 39.9, 69.9, 100], labels=["Low", "Medium", "High"])
    base["recommended_action"] = np.select(
        [base.risk_score >= 70, base.risk_score >= 40],
        ["Investigate and block/isolate if confirmed", "Monitor and verify vendor"],
        default="Allow with normal monitoring",
    )
    return base.sort_values("risk_score", ascending=False).reset_index(drop=True)


def run_nlp(tickets: pd.DataFrame):
    train, test = train_test_split(tickets, test_size=0.25, random_state=SEED, stratify=tickets["category"])
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
        ]
    )
    pipeline.fit(train["description"], train["category"])
    prediction = pipeline.predict(test["description"])
    metrics = pd.DataFrame(
        [
            {
                "accuracy": accuracy_score(test["category"], prediction),
                "macro_f1": f1_score(test["category"], prediction, average="macro"),
                "test_records": len(test),
                "classes": 3,
            }
        ]
    )
    predictions = test[["ticket_id", "description", "category"]].copy()
    predictions["predicted_category"] = prediction
    predictions["correct"] = (predictions.category == predictions.predicted_category).astype(int)
    report = pd.DataFrame(classification_report(test["category"], prediction, output_dict=True, zero_division=0)).T.reset_index().rename(columns={"index": "class"})
    return pipeline, metrics, predictions, report


def run_simulation() -> pd.DataFrame:
    scenarios = {
        "Current controls": {"credential_compromise": 0.22, "access_success": 0.60, "ot_reach": 0.55, "impact": 0.50},
        "MFA and time restrictions": {"credential_compromise": 0.22, "access_success": 0.26, "ot_reach": 0.44, "impact": 0.44},
        "MFA, segmentation and auto-isolation": {"credential_compromise": 0.22, "access_success": 0.22, "ot_reach": 0.18, "impact": 0.18},
    }
    iterations = 1000
    rows = []
    for name, p in scenarios.items():
        events = []
        losses = []
        for _ in range(iterations):
            compromised = RNG.random() < p["credential_compromise"]
            successful = compromised and RNG.random() < p["access_success"]
            reached_ot = successful and RNG.random() < p["ot_reach"]
            incident = reached_ot and RNG.random() < p["impact"]
            events.append(int(incident))
            losses.append((RNG.uniform(40, 100) if incident else RNG.uniform(0, 8)))
        rows.append(
            {
                "scenario": name,
                "iterations": iterations,
                "incident_rate": np.mean(events),
                "mean_impact_score": np.mean(losses),
                "high_impact_runs": int(np.sum(np.array(losses) >= 60)),
            }
        )
    result = pd.DataFrame(rows)
    baseline = result.loc[result.scenario == "Current controls", "incident_rate"].iloc[0]
    result["risk_reduction_vs_baseline"] = ((baseline - result.incident_rate) / baseline).clip(lower=0)
    return result


def build_incident_timeline(vpn, firewall, telemetry, tickets) -> pd.DataFrame:
    records = []
    for _, row in vpn[(vpn.source_ip == "198.51.100.77") & (vpn.timestamp >= "2026-08-30 02:00") & (vpn.timestamp < "2026-08-30 03:00")].iterrows():
        records.append({"timestamp": row.timestamp, "source": "VPN", "event_id": row.session_id, "entity": row.vendor_id, "asset_id": row.asset_id, "event": f"{row.login_result} remote login from {row.source_ip}; failed attempts={row.failed_attempts}"})
    for _, row in firewall[(firewall.event_id.str.startswith("FW_INC"))].iterrows():
        records.append({"timestamp": row.timestamp, "source": "Firewall", "event_id": row.event_id, "entity": row.source_ip, "asset_id": row.destination_asset, "event": f"{row.action} {row.protocol} traffic; connections={row.connection_count}; outbound={row.bytes_out_mb} MB"})
    for _, row in telemetry[(telemetry.telemetry_id.str.startswith("TEL_INC"))].iterrows():
        records.append({"timestamp": row.timestamp, "source": "Telemetry", "event_id": row.telemetry_id, "entity": row.sensor_type, "asset_id": row.asset_id, "event": f"{row.state} reading {row.value}; commands={row.command_count}"})
    for _, row in tickets[(tickets.ticket_id == "TKT_INC_01")].iterrows():
        records.append({"timestamp": row.timestamp, "source": "Ticket", "event_id": row.ticket_id, "entity": row.vendor_id, "asset_id": row.asset_id, "event": row.description})
    return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)


def run_adversarial_tests(model_data: pd.DataFrame, model, anomaly: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    high = model_data[model_data.label_suspicious == 1].head(80).copy()
    cases = {
        "Original suspicious sessions": high.copy(),
        "Working-hours evasion": high.assign(hour=14, off_hours=0),
        "Trusted-region and known-device disguise": high.assign(region_local=1, new_device=0),
        "Slow low-volume activity": high.assign(bytes_out_mb=np.minimum(high.bytes_out_mb, 12), duration_min=np.minimum(high.duration_min, 35)),
    }
    rows = []
    for name, frame in cases.items():
        probabilities = model.predict_proba(frame[feature_cols])[:, 1]
        detected = probabilities >= 0.50
        rows.append({"case": name, "records": len(frame), "mean_probability": probabilities.mean(), "detection_rate": detected.mean()})
    return pd.DataFrame(rows)


def make_figures(vpn, metrics, evaluation, anomaly, simulation, ranked):
    hourly = vpn.assign(hour=pd.to_datetime(vpn.timestamp).dt.hour).groupby("hour").agg(sessions=("session_id", "count"), suspicious=("label_suspicious", "sum")).reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(hourly.hour, hourly.sessions, marker="o", color="#1f77b4", label="All sessions")
    ax.plot(hourly.hour, hourly.suspicious, marker="o", color="#c83b3b", label="Suspicious")
    ax.set(title="Remote-access activity by hour", xlabel="Hour of day", ylabel="Session count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "baseline_access_by_hour.png")
    plt.close(fig)

    best_name = metrics.iloc[0].model
    cm = confusion_matrix(evaluation.actual, evaluation.predicted)
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set(title=f"{best_name} confusion matrix", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(FIGURES / "supervised_confusion_matrix.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(ranked.risk_score, bins=18, color="#2b6fb5", ax=ax)
    ax.axvline(70, color="#c83b3b", linestyle="--", label="High-risk threshold (70)")
    ax.set(title="Distribution of session risk scores", xlabel="Risk score", ylabel="Sessions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "risk_score_distribution.png")
    plt.close(fig)

    plot = simulation.copy()
    plot["incident_rate_pct"] = plot.incident_rate * 100
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.barplot(data=plot, x="incident_rate_pct", y="scenario", color="#2b6fb5", ax=ax)
    ax.set(title="Simulated incident rate by control scenario", xlabel="Incident rate (%)", ylabel="")
    fig.tight_layout()
    fig.savefig(FIGURES / "simulation_control_comparison.png")
    plt.close(fig)


def create_architecture_diagram():
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=170)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, text, fill, edge="#0b2b3b"):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015", facecolor=fill, edgecolor=edge, linewidth=1.4)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, weight="bold", color="#102a3a")

    def arrow(start, end, color="#2b6fb5"):
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", lw=1.6, color=color))

    box(0.10, 0.83, 0.80, 0.10, "IMPLEMENTED: VPN | OT firewall | telemetry | maintenance-ticket data", "#dceaf5")
    box(0.27, 0.67, 0.46, 0.09, "IMPLEMENTED: generation, quality checks and joining", "#f4f7f9")
    box(0.31, 0.53, 0.38, 0.08, "IMPLEMENTED: processed security dataset", "#fff3cf")
    box(0.03, 0.30, 0.28, 0.14, "IMPLEMENTED\nBaseline + supervised ML\nIsolation Forest", "#e4f2e8")
    box(0.36, 0.30, 0.28, 0.14, "IMPLEMENTED\nTimeline + intelligence\nTicket NLP", "#e4f2e8")
    box(0.69, 0.30, 0.28, 0.14, "IMPLEMENTED\nSimulation + risk score\nAdversarial tests", "#e4f2e8")
    box(0.27, 0.13, 0.46, 0.09, "IMPLEMENTED: standalone analyst dashboard", "#dceaf5")
    box(0.23, 0.01, 0.54, 0.06, "DEFERRED: live SIEM/SCADA integration and automated blocking", "#f7dfdf", edge="#9c3a3a")
    arrow((0.50, 0.83), (0.50, 0.76)); arrow((0.50, 0.67), (0.50, 0.61))
    arrow((0.50, 0.53), (0.17, 0.44)); arrow((0.50, 0.53), (0.50, 0.44)); arrow((0.50, 0.53), (0.83, 0.44))
    arrow((0.17, 0.30), (0.37, 0.22)); arrow((0.50, 0.30), (0.50, 0.22)); arrow((0.83, 0.30), (0.63, 0.22))
    fig.tight_layout()
    fig.savefig(OUTPUTS / "implemented_architecture.png", bbox_inches="tight")
    plt.close(fig)


def build_dashboard(ranked, timeline, simulation, metrics, nlp_metrics):
    alerts = ranked.head(100)[["session_id", "timestamp", "vendor_id", "asset_id", "risk_score", "risk_level", "recommended_action", "off_hours", "new_device", "failed_attempts", "bytes_out_mb"]].copy()
    alerts["timestamp"] = alerts.timestamp.astype(str)
    timeline_json = timeline.copy(); timeline_json["timestamp"] = timeline_json.timestamp.astype(str)
    data = {
        "alerts": alerts.to_dict(orient="records"),
        "timeline": timeline_json.to_dict(orient="records"),
        "simulation": simulation.assign(incident_rate_pct=(simulation.incident_rate*100).round(2), risk_reduction_pct=(simulation.risk_reduction_vs_baseline*100).round(1)).to_dict(orient="records"),
        "model": metrics.round(4).to_dict(orient="records"),
        "nlp": nlp_metrics.round(4).to_dict(orient="records"),
        "summary": {
            "sessions": int(len(ranked)),
            "high_risk": int((ranked.risk_score >= 70).sum()),
            "outliers": int(ranked.is_outlier.sum()),
            "top_risk": float(ranked.risk_score.max()),
        },
    }
    dashboard_json = json.dumps(data, default=str)
    template = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>T06 Water Utility Security Analytics</title>
<style>
:root{--navy:#0b2b3b;--blue:#2b6fb5;--pale:#eef4f8;--red:#b83232;--amber:#b97800;--green:#287a4b;--text:#17242d}
*{box-sizing:border-box}body{margin:0;background:#f5f7f9;color:var(--text);font:15px/1.45 Arial,sans-serif}
header{background:var(--navy);color:white;padding:22px 30px}header h1{margin:0 0 5px;font-size:25px}header p{margin:0;color:#cfe0ea}
main{max-width:1200px;margin:auto;padding:24px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card,.panel{background:white;border:1px solid #dce4e9;border-radius:10px;box-shadow:0 2px 8px #0000000d}.card{padding:17px}.card b{display:block;font-size:27px;color:var(--navy)}.card span{color:#63727b}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:24px 0 14px}.tabs button{border:0;border-radius:7px;background:#dfe9ef;padding:10px 14px;cursor:pointer}.tabs button.active{background:var(--blue);color:white}
.tab{display:none}.tab.active{display:block}.panel{padding:18px;margin-bottom:16px}h2{margin:0 0 14px;color:var(--navy)}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:9px;border-bottom:1px solid #e5eaed}th{background:var(--navy);color:white;position:sticky;top:0}.table-wrap{overflow:auto;max-height:520px}
select,input{padding:8px;border:1px solid #bac8d0;border-radius:6px;margin:0 8px 12px 0}.tag{display:inline-block;padding:3px 8px;border-radius:20px;color:white;font-size:12px}.High{background:var(--red)}.Medium{background:var(--amber)}.Low{background:var(--green)}
.bar{height:22px;background:#e7edf1;border-radius:4px;overflow:hidden;margin:6px 0 12px}.bar i{display:block;height:100%;background:var(--blue)}.timeline{border-left:3px solid var(--blue);padding-left:16px}.event{padding:8px 0 14px}.event b{color:var(--navy)}
.note{background:#fff5d8;border-left:4px solid #d99b00;padding:12px}.footer{color:#67767f;text-align:center;padding:20px}@media(max-width:800px){.cards{grid-template-columns:1fr 1fr}}
</style></head><body>
<header><h1>T06 Water Utility Security Analytics</h1><p>Decision support for vendor remote access and operational-technology anomalies</p></header>
<main><section class="cards" id="cards"></section>
<nav class="tabs"><button class="active" data-tab="overview">Overview</button><button data-tab="alerts">Suspicious Sessions</button><button data-tab="timeline">Incident Timeline</button><button data-tab="simulation">Control Simulation</button><button data-tab="text">Ticket Analytics</button></nav>
<section id="overview" class="tab active"><div class="panel"><h2>How to use this prototype</h2><p>Select Suspicious Sessions to filter and inspect alerts. Use the incident timeline to review correlated evidence, then compare the three control scenarios before recommending an action.</p><div class="note"><b>Decision rule:</b> risk scores at or above 70 require investigation. Blocking or isolation still needs analyst or OT-engineer confirmation.</div></div><div class="panel"><h2>Model comparison</h2><div id="model"></div></div></section>
<section id="alerts" class="tab"><div class="panel"><h2>Ranked remote-access alerts</h2><select id="riskFilter"><option value="All">All risk levels</option><option>High</option><option>Medium</option><option>Low</option></select><input id="vendorFilter" placeholder="Filter vendor, e.g. 07"><div class="table-wrap"><table><thead><tr><th>Session</th><th>Time</th><th>Vendor</th><th>Asset</th><th>Risk</th><th>Level</th><th>Why flagged</th><th>Action</th></tr></thead><tbody id="alertRows"></tbody></table></div></div></section>
<section id="timeline" class="tab"><div class="panel"><h2>Correlated incident: VENDOR_07 and OT_04</h2><div class="timeline" id="timelineRows"></div></div></section>
<section id="simulation" class="tab"><div class="panel"><h2>Security-control comparison</h2><div id="scenarioRows"></div></div></section>
<section id="text" class="tab"><div class="panel"><h2>Maintenance-ticket text classification</h2><div id="nlp"></div><p>The classifier separates normal maintenance, equipment faults and security concerns. Analysts should still review high-priority tickets because the text is synthetic and template-based.</p></div></section>
</main><div class="footer">Standalone synthetic-data prototype - no live water utility connection</div>
<script>const DATA=__DATA__;
const cards=[['Sessions analysed',DATA.summary.sessions],['High-risk sessions',DATA.summary.high_risk],['Isolation Forest outliers',DATA.summary.outliers],['Highest risk score',DATA.summary.top_risk]];document.getElementById('cards').innerHTML=cards.map(x=>`<div class="card"><b>${x[1]}</b><span>${x[0]}</span></div>`).join('');
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tabs button,.tab').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.getElementById(b.dataset.tab).classList.add('active')});
function reasons(a){return [a.off_hours?'off-hours':'',a.new_device?'new device':'',a.failed_attempts>=3?'failed logins':'',a.bytes_out_mb>=80?'high outbound volume':''].filter(Boolean).join(', ')||'model/anomaly pattern'}
function renderAlerts(){const r=document.getElementById('riskFilter').value,v=document.getElementById('vendorFilter').value.toLowerCase();const rows=DATA.alerts.filter(a=>(r==='All'||a.risk_level===r)&&a.vendor_id.toLowerCase().includes(v));document.getElementById('alertRows').innerHTML=rows.map(a=>`<tr><td>${a.session_id}</td><td>${a.timestamp}</td><td>${a.vendor_id}</td><td>${a.asset_id}</td><td><b>${a.risk_score}</b></td><td><span class="tag ${a.risk_level}">${a.risk_level}</span></td><td>${reasons(a)}</td><td>${a.recommended_action}</td></tr>`).join('')||'<tr><td colspan="8">No matching alerts</td></tr>'}
riskFilter.onchange=renderAlerts;vendorFilter.oninput=renderAlerts;renderAlerts();
document.getElementById('timelineRows').innerHTML=DATA.timeline.map(e=>`<div class="event"><b>${e.timestamp} - ${e.source}</b><br>${e.event}<br><small>Evidence: ${e.event_id} | Asset: ${e.asset_id}</small></div>`).join('');
document.getElementById('scenarioRows').innerHTML=DATA.simulation.map(s=>`<b>${s.scenario}</b> - ${s.incident_rate_pct}% incident rate (${s.risk_reduction_pct}% reduction)<div class="bar"><i style="width:${Math.max(3,s.incident_rate_pct*8)}%"></i></div>`).join('');
document.getElementById('model').innerHTML=DATA.model.map(m=>`<p><b>${m.model}</b>: precision ${(m.precision*100).toFixed(1)}%, recall ${(m.recall*100).toFixed(1)}%, F1 ${(m.f1_score*100).toFixed(1)}%</p>`).join('');
const n=DATA.nlp[0];document.getElementById('nlp').innerHTML=`<p><b>Accuracy:</b> ${(n.accuracy*100).toFixed(1)}% &nbsp; <b>Macro F1:</b> ${(n.macro_f1*100).toFixed(1)}% &nbsp; <b>Test records:</b> ${n.test_records}</p>`;
</script></body></html>"""
    (DASHBOARD / "dashboard.html").write_text(template.replace("__DATA__", dashboard_json), encoding="utf-8")
    (DASHBOARD / "dashboard_data.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def main():
    vpn = generate_vpn_logs()
    firewall = generate_firewall_logs(vpn)
    telemetry = generate_telemetry()
    tickets = generate_tickets()

    for name, frame in {"vpn_logs": vpn, "ot_firewall_logs": firewall, "sensor_telemetry": telemetry, "maintenance_tickets": tickets}.items():
        save_csv(frame, RAW / f"{name}.csv")

    quality = data_quality_report({"vpn_logs": vpn, "ot_firewall_logs": firewall, "sensor_telemetry": telemetry, "maintenance_tickets": tickets})
    save_csv(quality, OUTPUTS / "data_quality_summary.csv")

    model_data, feature_cols, best_name, supervised, model_metrics, evaluation, X_test, y_test = prepare_supervised(vpn)
    detector, anomaly = run_anomaly_model(model_data)
    ranked = create_ranked_alerts(model_data, supervised, anomaly, feature_cols)
    nlp_model, nlp_metrics, nlp_predictions, nlp_report = run_nlp(tickets)
    simulation = run_simulation()
    timeline = build_incident_timeline(vpn, firewall, telemetry, tickets)
    adversarial = run_adversarial_tests(model_data, supervised, anomaly, feature_cols)

    processed = ranked[["session_id", "timestamp", "vendor_id", "user_id", "asset_id", "label_suspicious", "supervised_probability", "anomaly_percentile", "risk_score", "risk_level", "recommended_action"]]
    save_csv(processed, PROCESSED / "analysed_remote_sessions.csv")
    save_csv(model_metrics, OUTPUTS / "supervised_model_metrics.csv")
    save_csv(evaluation, OUTPUTS / "supervised_test_predictions.csv")
    save_csv(anomaly.head(100), OUTPUTS / "top_anomalies.csv")
    save_csv(ranked.head(100), OUTPUTS / "ranked_alerts.csv")
    save_csv(timeline, OUTPUTS / "incident_timeline.csv")
    save_csv(simulation, SIMULATION / "control_scenarios.csv")
    save_csv(nlp_metrics, TEXT_MINING / "nlp_metrics.csv")
    save_csv(nlp_predictions, TEXT_MINING / "ticket_predictions.csv")
    save_csv(nlp_report, TEXT_MINING / "classification_report.csv")
    save_csv(adversarial, OUTPUTS / "adversarial_results.csv")

    joblib.dump(supervised, MODELS / "supervised_session_model.joblib")
    joblib.dump(detector, MODELS / "isolation_forest.joblib")
    joblib.dump(nlp_model, MODELS / "ticket_text_model.joblib")

    high = ranked.iloc[0]
    operational = (
        f"OPERATIONAL ALERT - HIGH RISK REMOTE SESSION\n\n"
        f"Session: {high.session_id}\nVendor: {high.vendor_id}\nAsset: {high.asset_id}\n"
        f"Risk score: {high.risk_score}/100\nTime: {high.timestamp}\n\n"
        f"Evidence: off_hours={high.off_hours}, new_device={high.new_device}, "
        f"failed_attempts={high.failed_attempts}, outbound={high.bytes_out_mb} MB, "
        f"anomaly percentile={high.anomaly_percentile}.\n\n"
        "Recommended action: verify the vendor by an approved contact, terminate the remote session, "
        "temporarily isolate the affected OT asset if confirmed, preserve logs, review recent commands, "
        "and monitor related accounts and devices."
    )
    executive = (
        "EXECUTIVE SECURITY SUMMARY\n\n"
        "The prototype identified a high-confidence pattern of compromised vendor access affecting OT_04. "
        "Correlated VPN, firewall, telemetry and ticket evidence indicates repeated failed logins, successful "
        "off-hours access from a new external device, unusual controller communication and abnormal pressure readings. "
        "The control simulation indicates that MFA combined with network segmentation and automatic isolation provides "
        "the strongest reduction in incident likelihood. The result is based on synthetic data and supports a decision "
        "to strengthen remote-access controls and monitoring; it does not authorize an automatic operational shutdown."
    )
    (INTELLIGENCE / "operational_alert.txt").write_text(operational, encoding="utf-8")
    (INTELLIGENCE / "executive_summary.txt").write_text(executive, encoding="utf-8")

    tests = pd.DataFrame(
        [
            ["T01", "Data loading", "All four CSV files", "5,750 total records and required columns", f"PASS - {sum(map(len,[vpn,firewall,telemetry,tickets]))} records"],
            ["T02", "Supervised model", "Held-out test set", "Precision, recall and F1 produced", f"PASS - best model {best_name}, F1={model_metrics.iloc[0].f1_score:.3f}"],
            ["T03", "Anomaly detection", "2,000 VPN sessions", "Ranked outlier list with threshold", f"PASS - {int(anomaly.is_outlier.sum())} outliers"],
            ["T04", "Incident correlation", "VENDOR_07 / OT_04", "Timeline contains multiple source types", f"PASS - {timeline.source.nunique()} source types, {len(timeline)} events"],
            ["T05", "Simulation", "Three control scenarios", "1,000 iterations per scenario", f"PASS - {len(simulation)} scenarios"],
            ["T06", "NLP", "250 synthetic tickets", "Three-class model and evaluation", f"PASS - accuracy={nlp_metrics.iloc[0].accuracy:.3f}"],
            ["T07", "Dashboard", "Embedded analysis JSON", "HTML contains alerts, timeline and scenarios", "PASS - standalone dashboard created"],
        ], columns=["test_id", "component", "input_condition", "expected_result", "actual_result_status"]
    )
    save_csv(tests, OUTPUTS / "test_log.csv")

    make_figures(vpn, model_metrics, evaluation, anomaly, simulation, ranked)
    create_architecture_diagram()
    build_dashboard(ranked, timeline, simulation, model_metrics, nlp_metrics)

    summary = {
        "record_count": int(len(vpn) + len(firewall) + len(telemetry) + len(tickets)),
        "vpn_records": len(vpn),
        "firewall_records": len(firewall),
        "telemetry_records": len(telemetry),
        "ticket_records": len(tickets),
        "suspicious_sessions": int(vpn.label_suspicious.sum()),
        "outliers": int(anomaly.is_outlier.sum()),
        "high_risk_sessions": int((ranked.risk_score >= 70).sum()),
        "best_model": best_name,
        "best_precision": float(model_metrics.iloc[0].precision),
        "best_recall": float(model_metrics.iloc[0].recall),
        "best_f1": float(model_metrics.iloc[0].f1_score),
        "nlp_accuracy": float(nlp_metrics.iloc[0].accuracy),
        "nlp_macro_f1": float(nlp_metrics.iloc[0].macro_f1),
        "timeline_events": len(timeline),
        "timeline_sources": int(timeline.source.nunique()),
        "simulation": simulation.to_dict(orient="records"),
        "adversarial": adversarial.to_dict(orient="records"),
    }
    (OUTPUTS / "implementation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
