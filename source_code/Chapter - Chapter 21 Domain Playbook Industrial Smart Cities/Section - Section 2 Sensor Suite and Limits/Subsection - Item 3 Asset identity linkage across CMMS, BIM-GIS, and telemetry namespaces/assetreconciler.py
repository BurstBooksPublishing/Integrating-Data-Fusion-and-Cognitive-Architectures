import uuid
import geopandas as gpd
from shapely.geometry import Point
from fuzzywuzzy import fuzz
import pandas as pd

# Load sources (assumed already normalized CRS)
cmms = pd.read_csv("cmms.csv")              # columns: asset_name, serial, cmms_id
bim = gpd.read_file("bim.geojson")         # columns: element_id, model, geometry
telemetry = pd.read_csv("telemetry.csv")   # columns: tag, lat, lon, last_seen

# Spatial index for BIM search
bim_sindex = bim.sindex

def attr_sim(a, b):
    # simple fuzzy average across fields
    s1 = fuzz.token_sort_ratio(str(a.get('asset_name','')), str(b.get('model','')))
    s2 = fuzz.partial_ratio(str(a.get('serial','')), str(b.get('serial','')))
    return 0.6*(s1/100.0) + 0.4*(s2/100.0)

def spatial_sim(tele_row, bim_row, sigma=10.0):
    p = Point(tele_row['lon'], tele_row['lat'])
    d = p.distance(bim_row.geometry)            # assumes metric CRS
    return float(np.exp(-d*d/(2*sigma*sigma)))

def reconcile_row(trow):
    # candidate BIM elements within 50m
    p = Point(trow['lon'], trow['lat'])
    candidates_idx = list(bim_sindex.intersection(p.buffer(50).bounds))
    best = None
    best_score = 0.0
    for i in candidates_idx:
        brow = bim.iloc[i]
        sa = attr_sim(trow, brow)
        ss = spatial_sim(trow, brow)
        S = 0.5*sa + 0.5*ss                      # simple weighting
        if S > best_score:
            best_score, best = S, brow
    if best_score > 0.65:
        return {'tele_tag': trow['tag'], 'bim_element': best.element_id,
                'score': best_score, 'mapping_id': str(uuid.uuid4())}
    return {'tele_tag': trow['tag'], 'bim_element': None, 'score': best_score, 'mapping_id': None}

# Reconcile telemetry rows (vectorize or map for production)
mappings = [reconcile_row(r) for _, r in telemetry.iterrows()]
mappings_df = pd.DataFrame(mappings)
mappings_df.to_csv("asset_mappings.csv", index=False)  # persistent mapping table