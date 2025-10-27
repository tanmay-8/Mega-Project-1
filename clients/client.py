from __future__ import annotations
import os
import time
from pathlib import Path

from clients.base_client import FLClient


def main():
    server_url = os.environ.get("SERVER_URL", "http://server:8000")
    data_csv = os.environ.get("DATA_CSV", "data/clients/client_1/train.csv")
    label_col = os.environ.get("LABEL_COL", "Class")
    rounds = int(os.environ.get("ROUNDS", "5"))
    local_epochs = int(os.environ.get("LOCAL_EPOCHS", "1"))
    batch_size = int(os.environ.get("BATCH_SIZE", "256"))
    lr = float(os.environ.get("LR", "0.1"))
    reg = float(os.environ.get("REG", "0.0"))
    delay = float(os.environ.get("DELAY", "0.0"))
    secagg = os.environ.get("SECAGG", "0") == "1"

    client = FLClient(server_url=server_url, data_csv=data_csv, label_col=label_col, lr=lr, reg=reg,
                      batch_size=batch_size, local_epochs=local_epochs, secagg=secagg)

    for r in range(rounds):
        if delay > 0:
            time.sleep(delay)
        res = client.train_one_round()
        print(f"Round {r+1}/{rounds} submit result: {res}")


if __name__ == "__main__":
    main()
