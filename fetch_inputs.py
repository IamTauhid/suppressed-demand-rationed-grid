"""
Stage 0 (network): fetch the two external inputs.  The PGCB record itself is included in
data/ as supplied (Mendeley Data doi:10.17632/vpk8spw2mm.1).

Writes  data/nasa_power_raw.json   NASA POWER daily, eight divisional HQs
        data/wdi_bgd.json          World Bank WDI, Bangladesh
"""
import json, os, time
import requests

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
DIV = {"Dhaka": (23.8103, 90.4125), "Chattogram": (22.3569, 91.7832), "Rajshahi": (24.3745, 88.6042),
       "Khulna": (22.8456, 89.5403), "Barishal": (22.7010, 90.3535), "Sylhet": (24.8949, 91.8687),
       "Rangpur": (25.7439, 89.2752), "Mymensingh": (24.7471, 90.4203)}


def main():
    out = {}
    for name, (lat, lon) in DIV.items():
        r = requests.get("https://power.larc.nasa.gov/api/temporal/daily/point",
                         params={"parameters": "T2M,RH2M,T2M_MAX,T2M_MIN", "community": "RE", "longitude": lon,
                                 "latitude": lat, "start": "20150401", "end": "20260308", "format": "JSON"}, timeout=180)
        r.raise_for_status(); out[name] = r.json()["properties"]["parameter"]; time.sleep(1)
    json.dump(out, open(os.path.join(DATA, "nasa_power_raw.json"), "w"))
    wdi = {}
    for ind in ["NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG", "NY.GDP.MKTP.KD"]:
        r = requests.get(f"https://api.worldbank.org/v2/country/BGD/indicator/{ind}",
                         params={"format": "json", "date": "2014:2026", "per_page": 50}, timeout=60)
        r.raise_for_status(); wdi[ind] = {x["date"]: x["value"] for x in r.json()[1]}
    json.dump(wdi, open(os.path.join(DATA, "wdi_bgd.json"), "w"))
    print("wrote data/nasa_power_raw.json, data/wdi_bgd.json")


if __name__ == "__main__":
    main()
