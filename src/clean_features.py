import pandas as pd

def main():
    expected_cols = [
        "location", "property_type", "bedrooms", "bathrooms",
        "size_sqm", "amenities", "price_kes", "listing_date", "source"
    ]

    df = pd.read_csv("data/raw_listings.csv", engine="python")

    # case: whole row got stuffed into 'location' column
    if df["property_type"].isna().all() and df["location"].astype(str).str.contains(",").any():
        split = df["location"].astype(str).str.strip().str.strip('"').str.split(",", n=8, expand=True)
        split.columns = expected_cols
        df = split
    else:
        df = df.iloc[:, :9]
        df.columns = expected_cols

    # Basic cleaning
    df["location"] = df["location"].astype(str).str.strip().str.title()
    df["property_type"] = df["property_type"].astype(str).str.strip().str.title()

    # numeric columns
    for col in ["bedrooms", "bathrooms", "size_sqm", "price_kes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # amenities formatting
    df["amenities"] = (
        df["amenities"].fillna("")
        .astype(str)
        .str.lower()
        .str.replace(",", "|", regex=False)
        .str.replace(";", "|", regex=False)
    )

    # remove duplicates
    df = df.drop_duplicates(subset=["source"])
    df = df.dropna(subset=["location", "price_kes"])

    # Feature engineering
    df["amenity_score"] = df["amenities"].apply(
        lambda x: len([a for a in str(x).split("|") if a.strip()])
    )

    df["price_per_sqm"] = df.apply(
        lambda r: (r["price_kes"] / r["size_sqm"])
        if pd.notna(r["size_sqm"]) and r["size_sqm"] > 0 else pd.NA,
        axis=1
    )

    # Month from listing_date
    dt = pd.to_datetime(df["listing_date"], errors="coerce")
    df["month"] = dt.dt.month

    out_path = "data/clean_listings.csv"
    df.to_csv(out_path, index=False)

    print(f"Clean file saved: {out_path}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")

if __name__ == "__main__":
    main()