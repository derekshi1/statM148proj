import polars as pl
import numpy as np

# Load raw journey data
DATA_PATH = "/Users/derek/UCLA_stats/statM148proj/data/training_new_data.parquet"
df = pl.read_parquet(DATA_PATH)

SUCCESS_ID = 28

# 1. Create label and extract TPP features
processed_df = df.with_columns([
    # Create the label: Does the journey list contain ID 28?
    pl.col("journey").list.eval(
        pl.element().struct.field("ed_id") == SUCCESS_ID
    ).list.any().alias("label_order_shipped")
]).select([
    pl.col("id"),
    pl.col("label_order_shipped"),
    # Sequence of event IDs
    pl.col("journey").list.eval(pl.element().struct.field("ed_id")).alias("event_seq"),
    # Sequence of time deltas (seconds between events)
    pl.col("journey").list.eval(
        (pl.element().struct.field("event_timestamp").diff().dt.total_seconds().fill_null(0))
    ).alias("delta_t_seq"),
    # Absolute time for intensity calculation
    pl.col("journey").list.eval(
        (pl.element().struct.field("event_timestamp") - pl.element().struct.field("event_timestamp").first()).dt.total_seconds()
    ).alias("abs_time_seq")
])

# 2. Filter for valid sequences (TPP needs at least two points)
processed_df = processed_df.filter(pl.col("event_seq").list.len() >= 2)

print(f"Processed {len(processed_df)} journeys.")
print(processed_df.head())

MAX_LEN = 128
def prepare_tpp_sequences(df, max_len):
    events_list = []
    deltas_list = []
    masks_list = []
    
    # Pre-convert to list for faster iteration
    event_seqs = df["event_seq"].to_list()
    delta_seqs = df["delta_t_seq"].to_list()
    
    for ev_seq, dt_seq in zip(event_seqs, delta_seqs):
        length = len(ev_seq)
        
        # Initialize zero-filled containers (0 is our Padding ID)
        e_padded = np.zeros(max_len, dtype=np.int64)
        d_padded = np.zeros(max_len, dtype=np.float32)
        m_padded = np.zeros(max_len, dtype=np.float32)
        
        # Apply Log-Scaling to Deltas immediately
        # np.log1p(x) is log(1 + x), essential for handling 0-second gaps
        log_deltas = np.log1p(dt_seq)
        
        if length > max_len:
            # Truncate: Keep MOST RECENT 128 events
            e_padded[:] = ev_seq[-max_len:]
            d_padded[:] = log_deltas[-max_len:]
            m_padded[:] = 1.0
        else:
            # Post-padding: Real data at start, zeros at end
            e_padded[:length] = ev_seq
            d_padded[:length] = log_deltas
            m_padded[:length] = 1.0
            
        events_list.append(e_padded)
        deltas_list.append(d_padded)
        masks_list.append(m_padded)
        
    return np.array(events_list), np.array(deltas_list), np.array(masks_list)

# 4. Generate Arrays
event_arr, delta_arr, mask_arr = prepare_tpp_sequences(processed_df, MAX_LEN)
labels = processed_df["label_order_shipped"].to_numpy()

SAVE_PATH = "/Users/derek/UCLA_stats/statM148proj/data/tpp_processed_data.npz"

np.savez_compressed(
    SAVE_PATH, 
    event_arr=event_arr, 
    delta_arr=delta_arr, 
    mask_arr=mask_arr, 
    labels=labels
)

print(f"Arrays saved successfully to {SAVE_PATH}")