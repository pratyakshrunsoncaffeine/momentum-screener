"""Public macro sources, immutable observation vintages and portable recovery."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
from datetime import date
from io import BytesIO, StringIO
from pathlib import Path
import hashlib
import json
import sqlite3
import ssl
import zipfile

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass(frozen=True)
class MacroSeries:
    identifier: str
    name: str
    family: str
    source: str
    frequency: str = "M"
    unit: str = "index"
    transform: str = "yoy"
    provider: str = "import"
    code: str = ""


FAMILIES = {
    "National accounts": ("https://mospi.gov.in/web/mospi/data", "Q", "Real GDP|Nominal GDP|PFCE real|GFCE real|GFCF real|GDP deflator|Agriculture GVA|Mining GVA|Manufacturing GVA|Utilities GVA|Construction GVA|Trade transport GVA|Financial services GVA|Public administration GVA|Exports national accounts|Imports national accounts|Inventories"),
    "Consumer inflation": ("https://mospi.gov.in/web/mospi/data", "M", "CPI headline|CPI rural|CPI urban|CPI food|CPI fuel|CPI housing|CPI clothing|CPI health|CPI transport|CPI education|CPI personal care|CPI cereals|CPI vegetables|CPI pulses|CPI milk|CPI edible oil|Core CPI"),
    "Producer inflation": ("https://eaindustry.nic.in/download_data_2223.asp", "M", "WPI headline|WPI primary|WPI food|WPI fuel|WPI manufactured|WPI metals|WPI chemicals|WPI textiles|WPI machinery|WPI food products|Output PPI|Input PPI"),
    "Industrial activity": ("https://mospi.gov.in/web/mospi/data", "M", "IIP total|IIP manufacturing|IIP mining|IIP electricity|IIP primary goods|IIP capital goods|IIP intermediate goods|IIP infrastructure|IIP consumer durables|IIP consumer non durables"),
    "Core industries": ("https://eaindustry.nic.in/", "M", "Core industries total|Coal output|Crude output|Natural gas output|Refinery output|Fertilizer output|Steel output|Cement output|Electricity output"),
    "Rates": ("https://data.rbi.org.in/", "M", "Repo rate|SDF rate|MSF rate|Bank rate|CRR|SLR|WACR|India 91D yield|India 182D yield|India 364D yield|India 2Y yield|India 5Y yield|India 10Y yield|Lending rate|Deposit rate|New loan rate"),
    "Liquidity": ("https://data.rbi.org.in/", "M", "Net liquidity injection|LAF outstanding|Repo injection|SDF absorption|VRR|VRRR|OMO purchases|OMO sales|Currency circulation|Reserve money|M1|M3|Bank reserves|Government cash"),
    "Credit": ("https://data.rbi.org.in/", "M", "Bank credit|Non food credit|Deposits|Agriculture credit|Industrial credit|Services credit|Personal credit|Housing credit|Vehicle credit|Consumer durable credit|Credit card outstanding|Education credit|Gold loans|NBFC credit|Commercial real estate credit|Infrastructure credit|Power credit|Road credit|Telecom credit|Metals credit|Textile credit|Chemical credit|Food processing credit|Construction credit|Engineering credit|Petroleum credit|MSME credit|Large industry credit"),
    "Fiscal": ("https://cga.gov.in/Page/Monthly-Accounts-Review.aspx", "M", "Government expenditure|Revenue expenditure|Government capex|Tax revenue|Non tax revenue|Receipts|Corporate tax|Income tax|Customs|Excise|Fiscal deficit|Revenue deficit|Primary deficit|Interest payments|Government borrowing|Road ministry capex|Railway capex|Defence capex"),
    "GST": ("https://www.gst.gov.in/", "M", "Gross GST|Net GST|Domestic GST|Import GST|CGST|SGST|IGST|Compensation cess"),
    "Trade": ("https://tradestat.commerce.gov.in/", "M", "Exports|Imports|Trade balance|Engineering exports|Electronics exports|Pharma exports|Chemical exports|Textile exports|Jewellery exports|Petroleum exports|Agricultural exports|Auto exports|Crude imports|Gold imports|Electronics imports|Machinery imports|Chemical imports|Coal imports|Fertilizer imports|Metal imports|Exports to US|Imports from China|Exports to EU|UAE trade|ASEAN trade"),
    "External": ("https://data.rbi.org.in/", "Q", "Current account balance|External debt|Short term external debt|FDI flows|Portfolio flows|ECB flows|Remittances|Services exports|Services imports"),
    "FX": ("https://data.rbi.org.in/", "M", "USD INR|EUR INR|GBP INR|JPY INR|NEER|REER|FX reserves|Foreign currency assets|Gold reserves|SDR|Reserve tranche"),
    "Petroleum": ("https://ppac.gov.in/index.php", "M", "Indian crude basket|Crude production|Crude import volume|Crude import value|Refinery throughput|Refinery utilisation|Petrol consumption|Diesel consumption|LPG consumption|ATF consumption|Naphtha consumption|Bitumen consumption|Petroleum consumption|Gas production|Gas consumption|LNG imports"),
    "Power": ("https://cea.nic.in/old/monthlyreports.html", "M", "Power generation|Power demand|Peak demand|Peak deficit|Energy deficit|Thermal generation|Hydro generation|Renewable generation|Nuclear generation|Plant load factor|Power plant coal stocks"),
    "Vehicles": ("https://analytics.parivahan.gov.in/analytics/vahanpublicreport?lang=en", "M", "Passenger vehicles|Two wheelers|Three wheelers|Commercial vehicles|Tractors|Buses|EV registrations|Petrol vehicles|Diesel vehicles|CNG vehicles"),
    "Labour": ("https://labourbureau.gov.in/dashboards", "M", "CPI IW|CPI AL|CPI RL|Rural wages|Agricultural wages|Construction wages|Unemployment|Rural unemployment|Urban unemployment|Labour participation|Worker population ratio|EPFO additions"),
    "Rural": ("https://imdpune.gov.in/lrfindex.php", "M", "Rainfall|Rainfall departure|Monsoon rainfall|Deficient districts|Reservoir storage|Kharif sowing|Rabi sowing|Rice acreage|Wheat acreage|Pulses acreage|Oilseed acreage|Cotton acreage|Sugarcane acreage|Fertilizer sales|Foodgrain production|Foodgrain stocks|Rice procurement|Wheat procurement|MSP"),
    "Food prices": ("https://consumeraffairs.nic.in/", "M", "Rice price|Wheat price|Atta price|Gram price|Tur price|Urad price|Moong price|Masoor price|Sugar price|Milk price|Groundnut oil price|Mustard oil price|Soybean oil price|Sunflower oil price|Palm oil price|Potato price|Onion price|Tomato price"),
    "Consumption and logistics": ("https://data.rbi.org.in/", "M", "Credit card spending|Debit card spending|UPI value|UPI volume|NEFT|Air passengers|Air cargo|Rail freight|Rail passengers|Port cargo|Container traffic"),
    "Surveys and housing": ("https://statistics.rbi.org.in/", "Q", "Consumer confidence|Rural confidence|Future expectations|Income expectations|Employment expectations|Inflation expectations 3M|Inflation expectations 1Y|Manufacturing outlook|Services outlook|Infrastructure outlook|Capacity utilisation|Order books|Inventory sales ratio|Credit standards|Loan demand|RBI house prices|NHB RESIDEX"),
    "Global": ("https://fred.stlouisfed.org/", "M", "US CPI|US core CPI|US PCE|US unemployment|US payrolls|US GDP|US industrial production|US Fed funds|US 2Y yield|US 10Y yield|US crude stocks|US oil production|China GDP|China industrial production|China retail sales|China PMI|China CPI|China PPI|China credit|China policy rate|Eurozone CPI|ECB deposit rate|Eurozone GDP|Japan policy rate|Japan CPI"),
}


def slug(name):
    return "_".join("".join(c.lower() if c.isalnum() else " " for c in name).split())


def catalogue():
    items = {}
    for family, (source, frequency, names) in FAMILIES.items():
        for name in names.split("|"):
            kind = "change" if family == "Rates" else "yoy"
            items[slug(name)] = MacroSeries(slug(name), name, family, source, frequency,
                                            "percent" if family == "Rates" else "source units", kind)
    # FRED is an official distributor; economic histories remain revised-only.
    fred = {
        "US CPI": ("CPIAUCSL", "yoy"), "US core CPI": ("CPILFESL", "yoy"),
        "US PCE": ("PCEPI", "yoy"), "US unemployment": ("UNRATE", "change"),
        "US payrolls": ("PAYEMS", "yoy"), "US industrial production": ("INDPRO", "yoy"),
        "US Fed funds": ("FEDFUNDS", "change"), "US 2Y yield": ("GS2", "change"),
        "US 10Y yield": ("GS10", "change"), "USD INR": ("EXINUS", "pct"),
        "India 10Y yield": ("IRLTLT01INM156N", "change"),
    }
    for name, (code, transform) in fred.items():
        old = items[slug(name)]
        items[old.identifier] = MacroSeries(old.identifier, name, old.family,
            "https://fred.stlouisfed.org/series/" + code, "M", "percent" if transform == "change" else "index", transform, "fred", code)
    for name, code in [("Brent crude", "DCOILBRENTEU"), ("WTI crude", "DCOILWTICO")]:
        items[slug(name)] = MacroSeries(slug(name), name, "Commodities",
            "https://fred.stlouisfed.org/series/" + code, "M", "USD/barrel", "pct", "fred_daily", code)
    for name, code in [("Brent crude", "Crude oil, Brent"), ("WTI crude", "Crude oil, WTI"), ("Gold", "Gold"), ("Copper", "Copper"), ("Aluminium", "Aluminum"),
                        ("Coal", "Coal, Australian"), ("Natural gas", "Natural gas, U.S."), ("Silver", "Silver")]:
        items[slug(name)] = MacroSeries(slug(name), name, "Commodities", "https://www.worldbank.org/en/research/commodity-markets", "M", "USD source units", "pct", "worldbank", code)
    for name, indicator in [("Real GDP", 5), ("Nominal GDP", 5), ("PFCE real", 10), ("GFCE real", 11), ("GFCF real", 9), ("Exports national accounts", 14), ("Imports national accounts", 15), ("Inventories", 12)]:
        old = items[slug(name)]
        spec = {"endpoint": "nas/getNASData", "params": {"base_year": "2011-12", "series": "Current", "frequency_code": "Quarterly", "indicator_code": indicator}, "value": "current_price" if name == "Nominal GDP" else "constant_price"}
        items[old.identifier] = MacroSeries(old.identifier, name, old.family, "https://api.mospi.gov.in/api/nas/getNASData", "Q", "INR crore", "yoy", "mospi", json.dumps(spec))
    for name, category in [("IIP total", 4), ("IIP mining", 1), ("IIP manufacturing", 2), ("IIP electricity", 3), ("IIP primary goods", 5), ("IIP capital goods", 6), ("IIP intermediate goods", 7), ("IIP infrastructure", 8), ("IIP consumer durables", 9), ("IIP consumer non durables", 10)]:
        old = items[slug(name)]
        spec = {"endpoint": "iip/getIIPMonthly", "params": {"base_year": "2011-12", "category_code": category, "type": "General" if category == 4 else "Sectoral" if category < 4 else "Use-based category"}, "value": "index", "empty": ["sub_category"]}
        items[old.identifier] = MacroSeries(old.identifier, name, old.family, "https://api.mospi.gov.in/api/iip/getIIPMonthly", "M", "index 2011-12=100", "yoy", "mospi", json.dumps(spec))
    for name, group, subgroup, sector in [("CPI headline", "0", "0.99", 3), ("CPI rural", "0", "0.99", 1), ("CPI urban", "0", "0.99", 2), ("CPI food", "1", "1.99", 3), ("CPI fuel", "5", "5.99", 3), ("CPI housing", "4", "4.99", 2), ("CPI clothing", "3", "3.99", 3), ("CPI cereals", "1", "1.1.01", 3), ("CPI vegetables", "1", "1.1.07", 3), ("CPI pulses", "1", "1.1.08", 3), ("CPI milk", "1", "1.1.04", 3), ("CPI edible oil", "1", "1.1.05", 3)]:
        old = items[slug(name)]
        spec = {"endpoint": "cpi/getCPIIndex", "params": {"base_year": "2012", "series": "Current", "state_code": 99, "sector_code": sector, "group_code": group, "subgroup_code": subgroup}, "value": "index"}
        items[old.identifier] = MacroSeries(old.identifier, name, old.family, "https://api.mospi.gov.in/api/cpi/getCPIIndex", "M", "index 2012=100", "yoy", "mospi", json.dumps(spec))
    for name, code in [("WPI headline", "1000000000"), ("WPI primary", "1100000000"), ("WPI fuel", "1200000000"), ("WPI manufactured", "1300000000"), ("WPI food", "2000000000")]:
        old = items[slug(name)]
        spec = {"endpoint": "wpi/getWpiRecords", "params": {"base_year": "2011-12", "major_group_code": code}, "value": "index_value", "empty": ["group", "subgroup", "sub_subgroup", "item"]}
        items[old.identifier] = MacroSeries(old.identifier, name, old.family, "https://api.mospi.gov.in/api/wpi/getWpiRecords", "M", "index 2011-12=100", "yoy", "mospi", json.dumps(spec))
    return items


CATALOGUE = catalogue()
COLUMNS = ["series_id", "period", "available_at", "retrieved_at", "value", "vintage", "eligible", "source", "base_year"]


def clean_observations(frame):
    f = frame.copy()
    if not set(COLUMNS).issubset(f):
        raise ValueError("Missing observation columns: " + ", ".join(sorted(set(COLUMNS) - set(f))))
    if not f.series_id.isin(CATALOGUE).all():
        raise ValueError("Unrecognized macro series identifier.")
    for col in ["period", "available_at", "retrieved_at"]:
        f[col] = pd.to_datetime(f[col], errors="coerce", utc=True).dt.tz_convert(None)
    f["value"] = pd.to_numeric(f.value, errors="coerce")
    if f[["period", "available_at", "retrieved_at", "value"]].isna().any().any() or not np.isfinite(f.value).all():
        raise ValueError("Dates and values must be valid and finite; no rows were imported.")
    if (f.available_at < f.period).any():
        raise ValueError("Availability cannot precede the observation period end.")
    if not f.eligible.isin([0, 1, False, True]).all():
        raise ValueError("eligible must be 0 or 1.")
    f["eligible"] = f.eligible.astype(int)
    if f.duplicated(["series_id", "period", "vintage"]).any():
        raise ValueError("Duplicate period/vintage rows; resolve them before importing.")
    for col in ["vintage", "source", "base_year"]:
        f[col] = f[col].fillna("").astype(str)
    if f.vintage.eq("").any() or f.source.eq("").any():
        raise ValueError("Source and vintage are required.")
    return f[COLUMNS].sort_values(["series_id", "period", "available_at"])


class MacroStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "macro.sqlite"
        with self.connect() as con:
            con.execute("CREATE TABLE IF NOT EXISTS observations (series_id TEXT, period TEXT, available_at TEXT, retrieved_at TEXT, value REAL, vintage TEXT, eligible INTEGER, source TEXT, base_year TEXT, PRIMARY KEY(series_id,period,vintage))")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def put(self, frame):
        f = clean_observations(frame)
        for col in ["period", "available_at", "retrieved_at"]:
            f[col] = f[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
        with self.connect() as con:
            con.executemany("INSERT OR IGNORE INTO observations VALUES (?,?,?,?,?,?,?,?,?)", f.itertuples(index=False, name=None))

    def get(self, identifier=None):
        with self.connect() as con:
            frame = pd.read_sql_query("SELECT * FROM observations" + (" WHERE series_id=?" if identifier else ""), con, params=[identifier] if identifier else [])
        for c in ["period", "available_at", "retrieved_at"]:
            frame[c] = pd.to_datetime(frame[c])
        return frame

    def coverage(self):
        obs = self.get()
        rows = []
        for definition in CATALOGUE.values():
            f = obs[obs.series_id == definition.identifier]
            rows.append({**asdict(definition), "Status": "Active" if len(f) else ("Not downloaded" if definition.provider != "import" else "Manual Import"),
                "Observations": f.period.nunique(), "First period": f.period.min(), "Last period": f.period.max(),
                "Last fetched": f.retrieved_at.max(), "Point-in-time rows": int(f.eligible.sum()),
                "Period age days": (pd.Timestamp.now().normalize()-f.period.max()).days if len(f) else None,
                "Freshness": "Stale history" if len(f) and (pd.Timestamp.now().normalize()-f.period.max()).days > (185 if definition.frequency == 'Q' else 75) else "Current" if len(f) else "Missing",
                "Base years": ", ".join(f.base_year.unique()) if len(f) else ""})
        return pd.DataFrame(rows)

    def export(self):
        target = BytesIO()
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("observations.csv", self.get().to_csv(index=False))
            for path in self.root.glob("macro_*.csv"):
                z.writestr(path.name, path.read_bytes())
            z.writestr("archive.json", json.dumps({"version": 1}))
        return target.getvalue()

    def restore(self, data):
        with zipfile.ZipFile(BytesIO(data)) as z:
            if sum(i.file_size for i in z.infolist()) > 100_000_000:
                raise ValueError("Archive exceeds the 100 MB uncompressed limit.")
            if json.loads(z.read("archive.json"))["version"] != 1:
                raise ValueError("Unsupported archive version.")
            obs = clean_observations(pd.read_csv(z.open("observations.csv")))
            frames = {}
            for name in z.namelist():
                if name.startswith("macro_") and name.endswith(".csv") and Path(name).name == name and "/" not in name and "\\" not in name:
                    frames[name] = pd.read_csv(z.open(name))
            self.put(obs)
            for name, f in frames.items():
                save_csv(f, self.root / name)


def save_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def public_session():
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=2, backoff_factor=.5, status_forcelist=[429, 500, 502, 503, 504])))
    return s


class MospiTLSAdapter(HTTPAdapter):
    """The public MoSPI server needs legacy negotiation; certificate checks stay enabled."""
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.options |= 0x4
        kwargs["ssl_context"] = context
        return super().init_poolmanager(*args, **kwargs)


def parse_mospi(rows, definition):
    spec = json.loads(definition.code)
    f = pd.DataFrame(rows)
    for column in spec.get("empty", []):
        if column in f:
            f = f[f[column].fillna("").eq("")]
    if f.empty:
        raise ValueError("No matching national aggregate returned.")
    if definition.frequency == "Q":
        q = f.quarter.str.extract(r"Q([1-4])", expand=False).astype(int)
        fy = f.year.str[:4].astype(int)
        period = pd.to_datetime({"year": fy+(q == 4).astype(int), "month": ((q*3+2)%12)+1, "day": 1}) + pd.offsets.MonthEnd(0)
    else:
        period = pd.to_datetime(f.year.astype(str)+"-"+f.month.astype(str), format="%Y-%B", errors="coerce") + pd.offsets.MonthEnd(0)
    series = pd.Series(pd.to_numeric(f[spec["value"]], errors="coerce").to_numpy(), index=pd.DatetimeIndex(period))
    if series.index.has_duplicates:
        raise ValueError("Ambiguous source dimensions: multiple values per period.")
    return series


class PublicMacroProvider:
    def __init__(self, store):
        self.store = store
        self.session = public_session()
        self.session.mount("https://api.mospi.gov.in/", MospiTLSAdapter(max_retries=1))
        self.workbook = None

    def fetch(self, definition, start, end):
        if definition.provider == "mospi":
            spec = json.loads(definition.code)
            records = []
            page = 1
            while True:
                r = self.session.get("https://api.mospi.gov.in/api/"+spec["endpoint"], params={**spec["params"], "Format": "JSON", "limit": 50, "page": page}, timeout=25)
                r.raise_for_status()
                payload = r.json()
                if not isinstance(payload.get("data"), list):
                    raise ValueError("MoSPI returned metadata or an error instead of observations.")
                records.extend(payload["data"])
                pages = int(payload.get("meta_data", {}).get("totalPages", 1))
                if page >= pages:
                    break
                if page >= 50:
                    raise ValueError("Source exceeded the bounded page limit.")
                page += 1
            series = parse_mospi(records, definition)
        elif definition.provider.startswith("fred"):
            r = self.session.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": definition.code}, headers={"User-Agent": requests.utils.default_user_agent()}, timeout=15)
            r.raise_for_status()
            f = pd.read_csv(StringIO(r.text))
            series = pd.Series(pd.to_numeric(f.iloc[:, 1], errors="coerce").to_numpy(), index=pd.to_datetime(f.iloc[:, 0]))
            if definition.provider == "fred_daily":
                series = series.resample("ME").mean()
            else:
                series.index = series.index.to_period("M").to_timestamp("M")
        elif definition.provider == "worldbank":
            if self.workbook is None:
                r = self.session.get("https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/related/CMO-Historical-Data-Monthly.xlsx", timeout=40)
                r.raise_for_status()
                self.workbook = pd.read_excel(BytesIO(r.content), sheet_name="Monthly Prices", header=4)
            f = self.workbook
            dates = pd.to_datetime(f.iloc[:, 0].astype(str).str.replace("M", "-", regex=False), errors="coerce")
            series = pd.Series(pd.to_numeric(f[definition.code], errors="coerce").to_numpy(), index=dates).dropna()
            series.index = series.index.to_period("M").to_timestamp("M")
        else:
            raise ValueError("An official file import is required for this series.")
        series = series.loc[~series.index.isna()].dropna().sort_index()
        series = series[(series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))]
        if series.empty:
            raise ValueError("Source returned no observations in the requested window.")
        now = pd.Timestamp.now("UTC").tz_localize(None)
        digest = hashlib.sha256(series.to_csv().encode()).hexdigest()[:16]
        raw = self.store.root / "raw"
        raw.mkdir(exist_ok=True)
        series.rename("value").to_csv(raw / f"{definition.identifier}_{digest}.csv")
        # Current downloads do not prove historic publication vintages.
        return pd.DataFrame({"series_id": definition.identifier, "period": series.index,
            "available_at": now, "retrieved_at": now, "value": series.to_numpy(),
            "vintage": digest, "eligible": 0, "source": definition.source,
            "base_year": json.loads(definition.code)["params"]["base_year"] if definition.provider == "mospi" else "source definition"})

    def refresh(self, identifiers, start, end, progress=None, full=False):
        health = []
        requests_path = self.store.root / "macro_download_requests.csv"
        completed = pd.read_csv(requests_path) if requests_path.exists() else pd.DataFrame(columns=["identifier", "start", "end", "fetched"])
        for i, identifier in enumerate(identifiers):
            d = CATALOGUE[identifier]
            if progress:
                progress(i, len(identifiers), d.name)
            cached = self.store.get(identifier)
            try:
                recent = not cached.empty and cached.retrieved_at.max().date() == date.today() and cached.period.min() <= pd.Timestamp(start) + pd.Timedelta(days=40)
                matches = completed[completed.identifier == identifier]
                if not cached.empty and not matches.empty:
                    last = matches.iloc[-1]
                    recent = recent or (str(last.fetched) == str(date.today()) and pd.Timestamp(last.start) <= pd.Timestamp(start) and pd.Timestamp(last.end) >= pd.Timestamp(end))
                if full or not recent:
                    self.store.put(self.fetch(d, start, end))
                    completed = pd.concat([completed[completed.identifier != identifier], pd.DataFrame([{
                        "identifier": identifier, "start": str(start), "end": str(end), "fetched": str(date.today())}])], ignore_index=True)
                    save_csv(completed, requests_path)
                status, message = "Loaded", "Saved data" if recent and not full else "Downloaded"
            except Exception as exc:
                status, message = ("Cached fallback" if not cached.empty else "Unavailable"), str(exc)[:300]
            health.append({"Indicator": d.name, "Status": status, "Message": message})
            save_csv(pd.DataFrame(health), self.store.root / "macro_health.csv")
        if progress:
            progress(len(identifiers), len(identifiers), "Macro downloads complete")
        return pd.DataFrame(health)
