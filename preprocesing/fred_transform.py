#
# cell_id: 2fCFQ1-lHJZK
# ============================================================
# ↔️ Terapkan Transformasi pada Data FRED (Differencing / Log Transform)
# ============================================================
import pandas as pd
import numpy as np
# Assuming fred_data and FRED_SERIES are available in the global scope or imported

def apply_fred_transformations(log_stream, fred_data, FRED_SERIES, fred_metadata=None):
    """
    Applies differencing or log transform to FRED data series.

    Args:
        log_stream (StringIO): Stream to write log messages.
        fred_data (pd.DataFrame): DataFrame with combined FRED data, indexed by release date.
                                   Columns are the desired names from FRED_SERIES.
        FRED_SERIES (dict): Dictionary mapping desired names to FRED series IDs.
        fred_metadata (list, optional): List of metadata dictionaries for FRED series.

    Returns:
        dict: Dictionary where keys are original FRED series names and values
              are DataFrames with the transformed series (column name ending in '_Transformed').
              Includes 'effective_until_next_release' column if available.
    """
    log_stream.write("\n[INFO] Menerapkan transformasi pada data FRED agar stasioner...\n") # Changed print
    transformed_fred_data = {}

    # Helper function to apply transformation
    def apply_transformation(series, series_name, freq):
        """Applies differencing or log transform based on series name and frequency."""
        # Simple heuristic: Apply log transform to index-like series, differencing to others
        # Adjust this logic based on actual series characteristics if needed
        # Check for keywords in series_name or title from metadata
        is_index_like = False
        if fred_metadata:
             for meta in fred_metadata:
                 if meta.get("series_id") == FRED_SERIES.get(series_name):
                      title = meta.get("title", "")
                      if "Index" in series_name or "WEI" in series_name or "Index" in title or "WEI" in title:
                          is_index_like = True
                          break
        # Fallback if metadata not available or doesn't match
        if not is_index_like and ("Index" in series_name or "WEI" in series_name):
             is_index_like = True


        if is_index_like:
            log_stream.write(f"  [INFO] Menerapkan Log Transform pada {series_name}\n") # Changed print
            # Add a small constant to handle zero or negative values if necessary
            transformed_series = np.log(series.clip(lower=1e-9))
        else:
            log_stream.write(f"  [INFO] Menerapkan Differencing pada {series_name}\n") # Changed print
            transformed_series = series.diff().dropna()

        # Optional: Apply seasonal differencing if frequency is known and applicable
        # For weekly/monthly data, seasonal differencing (e.g., .diff(periods=...)) might be needed
        # if "Weekly" in freq:
        #     transformed_series = transformed_series.diff(periods=52).dropna() # Weekly
        # elif "Monthly" in freq:
        #      transformed_series = transformed_series.diff(periods=12).dropna() # Monthly

        return transformed_series

    # Check if the input is a DataFrame as expected from download_macro_data
    if isinstance(fred_data, pd.DataFrame) and not fred_data.empty:
        log_stream.write("  [INFO] Processing FRED data (DataFrame format)...\n") # Changed print

        # Iterate through the columns of the combined FRED DataFrame (excluding date/effective columns)
        value_cols = [col for col in fred_data.columns if col not in ["release_date", "effective_until_next_release", "date"]]

        if not value_cols:
             log_stream.write("  [WARN] Tidak ada kolom nilai yang terdeteksi dalam DataFrame FRED. Melewati transformasi.\n") # Changed print
             return transformed_fred_data # Return empty dict


        for name in value_cols:
            if name not in fred_data.columns:
                 log_stream.write(f"  [WARN] Kolom '{name}' tidak ditemukan dalam DataFrame. Melewati transformasi.\n") # Changed print
                 continue

            series_id = FRED_SERIES.get(name) # Get series_id from FRED_SERIES dict

            if fred_data[name].dropna().empty:
                 log_stream.write(f"  [WARN] Seri '{name}' kosong atau hanya NaN setelah dropna. Melewati transformasi.\n") # Changed print
                 transformed_fred_data[name] = pd.DataFrame() # Store empty DataFrame
                 continue

            try:
                # Get metadata to infer frequency if available
                meta = None
                if series_id and fred_metadata:
                     for m in fred_metadata:
                         if m.get("series_id") == series_id:
                             meta = m
                             break
                freq = meta.get("frequency", "Unknown") if meta else "Unknown"

                # Apply transformation to the series
                transformed_series = apply_transformation(fred_data[name].dropna(), name, freq) # Apply transformation to non-NaN values

                if transformed_series is not None and not transformed_series.empty:
                     # Create a new DataFrame for the transformed series, keeping index
                     transformed_df = transformed_series.to_frame(name=f"{name}_Transformed")
                     # Keep the effective_until_next_release column if it exists and align indices
                     if "effective_until_next_release" in fred_data.columns:
                         # Need to align the effective dates with the transformed series index
                         # Use reindex with forward fill to propagate the last known effective date
                         # Ensure both indices are timezone-aware before reindexing
                         fred_data_index_utc = pd.to_datetime(fred_data.index, errors='coerce', utc=True)
                         transformed_df_index_utc = pd.to_datetime(transformed_df.index, errors='coerce', utc=True)

                         if not fred_data_index_utc.isna().all() and not transformed_df_index_utc.isna().all():
                              effective_dates_aligned = fred_data["effective_until_next_release"].reindex(transformed_df_index_utc, method='ffill')
                              transformed_df["effective_until_next_release"] = effective_dates_aligned.reindex(transformed_series.index) # Reindex back to original transformed index name


                     transformed_fred_data[name] = transformed_df
                     log_stream.write(f"  [OK] {name}: Transformasi berhasil ({len(transformed_df)} observasi)\n") # Changed print

                else:
                     log_stream.write(f"  [WARN] Transformasi menghasilkan data kosong untuk {name}.\n") # Changed print
                     transformed_fred_data[name] = pd.DataFrame() # Store empty DataFrame

            except Exception as e:
                log_stream.write(f"  [ERROR] Gagal menerapkan transformasi untuk {name}: {e}\n") # Changed print
                transformed_fred_data[name] = pd.DataFrame() # Store empty DataFrame on failure

    else:
        log_stream.write("[WARN] Variabel 'fred_data' bukan DataFrame atau kosong.\n") # Changed print


    log_stream.write("\n[OK] Proses transformasi data FRED selesai.\n") # Changed print
    return transformed_fred_data
