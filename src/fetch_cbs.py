"""
Official CBS (Statistics Netherlands) municipality-level key figures for
Berg en Dal, via PDOK's "wijkenbuurten" WFS -- population, households,
housing stock, energy transition indicators (solar rooftop adoption, gas-free
homes), and business counts by sector. Not remote sensing -- this is the
official-registry half of the picture that satellite data alone can't give
you (nobody derives "number of households" from a satellite image), used
here as ground truth to sit next to what the satellite/LiDAR layers show.

Source: https://service.pdok.nl/cbs/wijkenbuurten/2024/wfs/v1_0
"""
from pathlib import Path

import requests

from statsutil import update_stats

WFS_URL = "https://service.pdok.nl/cbs/wijkenbuurten/2024/wfs/v1_0"
GEMEENTECODE = "GM1945"

# properties -> (english label, unit). Originally just the ~18 fields the
# dashboard used; extended to most of the non-suppressed fields the same
# WFS record actually carries (many CBS fields come back -99997 --
# "suppressed" -- below a population threshold; Berg en Dal is big enough
# that most of these aren't, so there's real extra municipal detail sitting
# in a response this pipeline was already making and only partly reading).
FIELDS = {
    "aantalInwoners": ("population", "residents"),
    "mannen": ("population_male", "residents"),
    "vrouwen": ("population_female", "residents"),
    "aantalHuishoudens": ("households", "households"),
    "gemiddeldeHuishoudsgrootte": ("avg_household_size", "people/household"),
    "percentageEenpersoonshuishoudens": ("single_person_households_pct", "%"),
    "percentageHuishoudensMetKinderen": ("households_with_children_pct", "%"),
    "woningvoorraad": ("housing_stock", "homes"),
    "aantalNieuwbouwWoningen": ("new_homes_last_year", "homes/yr"),
    "bevolkingsdichtheidInwonersPerKm2": ("population_density", "residents/km2"),
    "oppervlakteLandInHa": ("land_area_ha", "ha"),
    "oppervlakteWaterInHa": ("water_area_ha", "ha"),
    "percentageWoningenMetZonnestroom": ("homes_with_solar_pct", "%"),
    "percentageAardgasvrijeWoningen": ("gas_free_homes_pct", "%"),
    "percentageAardgaswoningen": ("gas_heated_homes_pct", "%"),
    "gemiddeldAardgasverbruik": ("avg_gas_use_m3", "m3/yr"),
    "gemiddeldeElektriciteitslevering": ("avg_electricity_use_kwh", "kWh/yr"),
    "gemiddeldeElektriciteitsteruglevering": ("avg_solar_feedback_kwh", "kWh/yr"),
    "gemiddeldeWoningwaarde": ("avg_home_value_x1000eur", "x1,000 EUR"),
    "percentageKoopwoningen": ("owner_occupied_homes_pct", "%"),
    "percentageHuurwoningen": ("rented_homes_pct", "%"),
    "percentageEengezinswoning": ("single_family_homes_pct", "%"),
    "percentageMeergezinswoning": ("multi_family_homes_pct", "%"),
    "aantalBedrijfsvestigingen": ("total_businesses", "businesses"),
    "aantalBedrijvenLandbouwBosbouwVisserij": ("farm_forestry_fishery_businesses", "businesses"),
    "aantalBedrijvenNijverheidEnergie": ("industry_energy_businesses", "businesses"),
    "aantalBedrijvenHandelEnHoreca": ("trade_hospitality_businesses", "businesses"),
    "aantalBedrijvenVervoerInformatieCommunicatie": ("transport_ict_businesses", "businesses"),
    "aantalBedrijvenFinancieelOnroerendGoed": ("finance_realestate_businesses", "businesses"),
    "aantalBedrijvenZakelijkeDienstverlening": ("business_services_businesses", "businesses"),
    "aantalBedrijvenOverheidOnderwijsEnZorg": ("government_education_care_businesses", "businesses"),
    "aantalBedrijvenCultuurRecreatieOverige": ("culture_recreation_other_businesses", "businesses"),
    "percentagePersonen0Tot15Jaar": ("population_under15_pct", "%"),
    "percentagePersonen15Tot25Jaar": ("population_15to25_pct", "%"),
    "percentagePersonen25Tot45Jaar": ("population_25to45_pct", "%"),
    "percentagePersonen45Tot65Jaar": ("population_45to65_pct", "%"),
    "percentagePersonen65JaarEnOuder": ("population_65plus_pct", "%"),
    "geboorteTotaal": ("births_last_year", "births/yr"),
    "sterfteTotaal": ("deaths_last_year", "deaths/yr"),
    "aantalPersonenMetEenAowUitkeringTotaal": ("aow_pension_recipients", "residents"),
    "aantalPersonenMetEenWwUitkeringTotaal": ("unemployment_benefit_recipients", "residents"),
    "aantalPersonenMetEenAlgBijstandsuitkeringTot": ("welfare_benefit_recipients", "residents"),
}


def fetch() -> dict:
    params = {
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeName": "wijkenbuurten:gemeenten", "outputFormat": "application/json",
    }
    r = requests.get(WFS_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    match = next(f for f in data["features"] if f["properties"]["gemeentecode"] == GEMEENTECODE)
    raw = match["properties"]

    out = {"year": raw.get("jaar"), "gemeentecode": GEMEENTECODE, "gemeentenaam": raw.get("gemeentenaam")}
    for cbs_key, (label, unit) in FIELDS.items():
        val = raw.get(cbs_key)
        out[label] = {"value": None if val == -99997 or val == -99997.0 else val, "unit": unit}

    update_stats("cbs", out)
    print(f"CBS {raw.get('gemeentenaam')} ({raw.get('jaar')}): "
          f"{out['population']['value']:,} residents, {out['households']['value']:,} households, "
          f"{out['homes_with_solar_pct']['value']}% homes with solar")
    return out


if __name__ == "__main__":
    fetch()
