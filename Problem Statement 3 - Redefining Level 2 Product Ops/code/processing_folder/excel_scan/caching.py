from excel_scanner import ensure_cache, DEFAULT_XLSX, DEFAULT_CACHE

if __name__ == "__main__":
    # Build (or refresh) the full Excel cache — no row limit
    cache = ensure_cache(
        xlsx_path=DEFAULT_XLSX,
        cache_path=DEFAULT_CACHE,
        rows_limit=None,       #  full Excel
        read_all_sheets=False  # True if multiple sheets
    )

    print(f" Cached {len(cache.items)} total cells from full Excel.")
