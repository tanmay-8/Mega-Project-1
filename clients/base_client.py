#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, List
import time
import csv
import json
import resource

import requests

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.logistic_from_scratch import LogisticRegressionScratch, load_csv_xy
from server.secure_aggregation.ecdh import generate_keypair
from server.secure_aggregation.bonawitz import make_mask_vector
from scripts.encoding import encode_vector_to_int
import base64


class FLClient:
    def __init__(self, server_url: str, data_csv: str, label_col: str = "Class", lr: float = 0.1, reg: float = 0.0,
                 batch_size: int = 256, local_epochs: int = 1, secagg: bool = False):
        self.server_url = server_url.rstrip("/")
        self.data_csv = data_csv
        self.label_col = label_col
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.X, self.y, self.feats = load_csv_xy(data_csv, label_col=label_col)
        self.model = LogisticRegressionScratch(n_features=len(self.feats), lr=self.lr, reg=self.reg)
        self.secagg = secagg
        self.Q = 2**61 - 1
        self.S = 2**16

    def fetch_global(self) -> Tuple[List[float], float, int]:
        r = requests.get(f"{self.server_url}/global_model", timeout=30)
        r.raise_for_status()
        js = r.json()
        return js["weights"], float(js["bias"]), int(js.get("round", 0))

    def submit_update(self, delta_w: List[float], delta_b: float) -> dict:
        payload = {"delta_w": delta_w, "delta_b": delta_b, "num_samples": len(self.X)}
        r = requests.post(f"{self.server_url}/submit_update", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def train_one_round(self) -> dict:
        gw, gb, rnd = self.fetch_global()
        # set global
        self.model.set_params(gw, gb)
        # local train
        # CPU before
        ru0 = resource.getrusage(resource.RUSAGE_SELF)
        cpu0 = float(ru0.ru_utime + ru0.ru_stime)
        t0 = time.time()
        self.model.fit(self.X, self.y, epochs=self.local_epochs, batch_size=self.batch_size, verbose=False)
        train_time = time.time() - t0
        ru1 = resource.getrusage(resource.RUSAGE_SELF)
        cpu_time = float(ru1.ru_utime + ru1.ru_stime) - cpu0
        # delta
        lw, lb = self.model.get_params()
        dw = [lw[i] - gw[i] for i in range(len(gw))]
        db = lb - gb
        if not self.secagg:
            # approx JSON bytes
            payload = {"delta_w": dw, "delta_b": db, "num_samples": len(self.X)}
            approx_bytes = len(json.dumps(payload).encode("utf-8"))
            res = self.submit_update(dw, db)
            self._log_client_perf(round_id=rnd, secagg=0, train_time=train_time, cpu_time=cpu_time, upload_bytes=approx_bytes)
            return res
        else:
            # SecAgg: register/get peers
            kp = generate_keypair()
            reg = requests.post(f"{self.server_url}/register_round", json={
                'client_id': os.environ.get('CLIENT_ID', 'client'),
                'pubkey': base64.b64encode(kp.serialize_public()).decode(),
            }, timeout=30)
            reg.raise_for_status()
            js = reg.json()
            # wait ready
            tries = 0
            while js.get('status') != 'ready' and tries < 60:
                import time
                time.sleep(1)
                reg = requests.post(f"{self.server_url}/register_round", json={
                    'client_id': os.environ.get('CLIENT_ID', 'client'),
                    'pubkey': base64.b64encode(kp.serialize_public()).decode(),
                }, timeout=30)
                js = reg.json()
                tries += 1
            if js.get('status') != 'ready':
                raise RuntimeError('SecAgg peers not ready')
            round_id = int(js['round_id'])
            # peers map
            peers = {cid: base64.b64decode(b64) for cid, b64 in js['pubkeys'].items()}
            my_id = os.environ.get('CLIENT_ID', 'client')
            mask = make_mask_vector(kp, peers, round_id, dim=len(dw), q=self.Q, my_id=my_id)
            enc = encode_vector_to_int(dw, S=self.S, q=self.Q)
            masked = [(enc[i] + mask[i]) % self.Q for i in range(len(enc))]
            payload = {"masked": masked, "num_samples": len(self.X)}
            rr = requests.post(f"{self.server_url}/submit_masked_update", json=payload, timeout=60)
            rr.raise_for_status()
            # approx bytes
            approx_bytes = 8 * len(masked)
            self._log_client_perf(round_id=round_id, secagg=1, train_time=train_time, cpu_time=cpu_time, upload_bytes=approx_bytes)
            return rr.json()

    def _log_client_perf(self, round_id: int, secagg: int, train_time: float, cpu_time: float, upload_bytes: int):
        """Append per-round client metrics."""
        state_dir = Path(os.environ.get("STATE_DIR", "server/state"))
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / "client_perf.csv"
        exists = path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["client_id", "round", "secagg", "local_train_time_s", "cpu_time_s", "upload_bytes"])
            w.writerow([os.environ.get("CLIENT_ID", "client"), round_id, secagg, f"{train_time:.6f}", f"{cpu_time:.6f}", upload_bytes])
