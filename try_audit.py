import pandas as pd
from rawaj.audit import audit
df = pd.read_csv("data/sample_transfers.csv")
print(audit(df, target="is_mule", id_col="txn_id", time_col="sent_at"))