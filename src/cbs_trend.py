"""
Population and housing stock, 2018-2024: how much Berg en Dal actually
grew, from CBS's own registry -- not modelled, the real yearly count.

PDOK republishes this same "wijkenbuurten" WFS (fetch_cbs.py) per
year-of-publication; checked directly which years actually resolve
(2018, 2019, 2022, 2023, 2024 return real data -- 2020 and 2021 don't
exist as separate services, a real gap in the source, not something
skipped here).

    python cbs_trend.py
"""
import requests

from statsutil import update_stats

GEMEENTECODE = "GM1945"
YEARS = [2018, 2019, 2022, 2023, 2024]  # checked directly -- 2020/2021 don't resolve as a WFS year


def _fetch_year(year: int) -> dict | None:
    """Checked directly: most years' WFS exposes a plain "gemeenten"
    type name, but 2018/2019 use a year-suffixed "gemeenten{year}"
    instead -- try both rather than skip those years."""
    url = f"https://service.pdok.nl/cbs/wijkenbuurten/{year}/wfs/v1_0"
    for type_name in ("wijkenbuurten:gemeenten", f"wijkenbuurten:gemeenten{year}"):
        try:
            r = requests.get(url, params={
                "service": "WFS", "version": "2.0.0", "request": "GetFeature",
                "typeName": type_name, "outputFormat": "application/json",
            }, timeout=30)
        except requests.RequestException:
            continue
        if r.status_code != 200:
            continue
        data = r.json()
        match = next((f for f in data["features"] if f["properties"]["gemeentecode"] == GEMEENTECODE), None)
        if match:
            return match["properties"]
    return None


def run() -> dict:
    years, population, households, housing_stock = [], [], [], []
    for year in YEARS:
        props = _fetch_year(year)
        if props is None:
            print(f"[cbs trend] {year}: not available, skipping")
            continue
        pop = props.get("aantalInwoners")
        hh = props.get("aantalHuishoudens")
        stock = props.get("woningvoorraad")
        years.append(year)
        population.append(None if pop == -99997 else pop)
        households.append(None if hh == -99997 else hh)
        housing_stock.append(None if stock == -99997 else stock)
        print(f"[cbs trend] {year}: {pop:,} residents, {hh:,} households, {stock:,} homes")

    if len(years) >= 2:
        i0, i1 = 0, -1
        net_pop = population[i1] - population[i0] if population[i0] and population[i1] else None
        net_stock = housing_stock[i1] - housing_stock[i0] if housing_stock[i0] and housing_stock[i1] else None
        print(f"[cbs trend] {years[i0]}->{years[i1]}: "
              f"population {net_pop:+,} residents, housing stock {net_stock:+,} homes"
              if net_pop is not None and net_stock is not None else "[cbs trend] change unavailable")

    update_stats("cbs_trend", {
        "years": years, "population": population, "households": households, "housing_stock": housing_stock,
    })
    return {"years": years, "population": population, "households": households, "housing_stock": housing_stock}


if __name__ == "__main__":
    run()
