import pandas as pd

def main():
    # 1) Read raw file
    import pandas as pd

def main():
    expected_cols = [
        "location", "property_type", "bedrooms", "bathrooms",
        "size_sqm", "amenities", "price_kes", "listing_date", "source"
    ]

    df = pd.read_csv("data/raw_listings.csv", engine="python")

    # ✅ CASE: whole row got stuffed into 'location' column (your current situation)
    if df["property_type"].isna().all() and df["location"].astype(str).str.contains(",").any():
        split = df["location"].astype(str).str.strip().str.strip('"').str.split(",", n=8, expand=True)
        split.columns = expected_cols
        df = split
    else:
        # normal case: ensure correct columns
        df = df.iloc[:, :9]
        df.columns = expected_cols


    # 3) Basic cleaning
    df["location"] = df["location"].astype(str).str.strip().str.title()
    df["property_type"] = df["property_type"].astype(str).str.strip().str.title()

    # numeric columns
    for col in ["bedrooms", "bathrooms", "size_sqm", "price_kes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # amenities formatting: use | separator, lowercase
    df["amenities"] = (
        df["amenities"].fillna("")
        .astype(str)
        .str.lower()
        .str.replace(",", "|", regex=False)
        .str.replace(";", "|", regex=False)
    )

    # remove duplicates by source
    df = df.drop_duplicates(subset=["source"])

    # drop rows missing core info
    df = df.dropna(subset=["location", "price_kes"])

    # 4) Feature engineering
    df["amenity_score"] = df["amenities"].apply(
        lambda x: len([a for a in str(x).split("|") if a.strip()])
    )

    df["price_per_sqm"] = df.apply(
        lambda r: (r["price_kes"] / r["size_sqm"])
        if pd.notna(r["size_sqm"]) and r["size_sqm"] > 0 else pd.NA,
        axis=1
    )

    # month from listing_date (optional)
    df["listing_date"] = df["listing_date"].astype(str).str.strip()
    dt = pd.to_datetime(df["listing_date"], errors="coerce")
    df["month"] = dt.dt.month

    # 5) Save clean file
    out_path = "data/clean_listings.csv"
    df.to_csv(out_path, index=False)

    print(f"✅ Clean file saved: {out_path}")
    print(f"✅ Rows: {len(df)} | Columns: {len(df.columns)}")
    print(df.head(3))

if __name__ == "__main__":
    main()
