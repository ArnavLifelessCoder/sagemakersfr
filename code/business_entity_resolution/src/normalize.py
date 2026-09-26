"""Text normalisation for business names and addresses.

Everything here is rule-based and uses only the provided data plus generic
linguistic knowledge (abbreviations, state names). No external lookups.
"""
import re
import unicodedata

from rapidfuzz.distance import JaroWinkler, Levenshtein
from unidecode import unidecode

# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
NON_LATIN_RE = re.compile(r"[ऀ-෿]")  # Devanagari .. Sinhala (all Indic blocks)
NULL_TOKENS = {"null", "<null>", "n/a", "na", "none", "nan", "-"}


def is_native(s):
    return bool(NON_LATIN_RE.search(s))


def fold(s):
    """NFKC + transliterate to ASCII + lowercase."""
    return unidecode(unicodedata.normalize("NFKC", s)).lower()


# ----------------------------------------------------------------------------
# States / regions (canonical codes). Keys are folded forms.
# ----------------------------------------------------------------------------
_US = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas", "CA": "california",
    "CO": "colorado", "CT": "connecticut", "DE": "delaware", "DC": "district of columbia",
    "FL": "florida", "GA": "georgia", "HI": "hawaii", "ID": "idaho", "IL": "illinois",
    "IN": "indiana", "IA": "iowa", "KS": "kansas", "KY": "kentucky", "LA": "louisiana",
    "ME": "maine", "MD": "maryland", "MA": "massachusetts", "MI": "michigan", "MN": "minnesota",
    "MS": "mississippi", "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada",
    "NH": "new hampshire", "NJ": "new jersey", "NM": "new mexico", "NY": "new york",
    "NC": "north carolina", "ND": "north dakota", "OH": "ohio", "OK": "oklahoma", "OR": "oregon",
    "PA": "pennsylvania", "RI": "rhode island", "SC": "south carolina", "SD": "south dakota",
    "TN": "tennessee", "TX": "texas", "UT": "utah", "VT": "vermont", "VA": "virginia",
    "WA": "washington", "WV": "west virginia", "WI": "wisconsin", "WY": "wyoming",
    "PR": "puerto rico",
}
_IN = {
    "MH": ["maharashtra", "महाराष्ट्र"], "DL": ["delhi", "nct of delhi", "दिल्ली"],
    "UP": ["uttar pradesh", "उत्तर प्रदेश"], "KA": ["karnataka", "ಕರ್ನಾಟಕ"],
    "TN": ["tamil nadu", "தமிழ்நாடு"], "GJ": ["gujarat", "ગુજરાત"],
    "WB": ["west bengal", "পশ্চিমবঙ্গ"], "TG": ["telangana", "తెలంగాణ"],
    "HR": ["haryana", "हरियाणा"], "RJ": ["rajasthan", "राजस्थान"],
    "KL": ["kerala", "keralam", "കേരളം"], "BR": ["bihar", "बिहार"],
    "MP": ["madhya pradesh", "मध्य प्रदेश"], "AP": ["andhra pradesh", "ఆంధ్రప్రదేశ్"],
    "PB": ["punjab", "ਪੰਜਾਬ"], "OD": ["odisha", "orissa", "ଓଡ଼ିଶା"],
    "AS": ["assam", "অসম"], "JH": ["jharkhand", "झारखंड"], "CG": ["chhattisgarh", "छत्तीसगढ़"],
    "UK": ["uttarakhand", "उत्तराखंड", "uttaranchal"], "HP": ["himachal pradesh", "हिमाचल प्रदेश"],
    "GA": ["goa", "गोवा"], "JK": ["jammu and kashmir", "jammu & kashmir"],
    "PY": ["puducherry", "pondicherry"],
}
# France: regions and departements seen in the data, both mapped to the region.
_FR = {
    "HDF": ["hauts-de-france", "hauts de france", "nord", "pas-de-calais", "pas de calais"],
    "NAQ": ["nouvelle-aquitaine", "nouvelle aquitaine", "gironde"],
    "PDL": ["pays de la loire", "loire-atlantique", "loire atlantique"],
}


def _build_state_map():
    m = {}
    for code, name in _US.items():
        m[("US", code.lower())] = "US-" + code
        m[("US", name)] = "US-" + code
    for code, names in _IN.items():
        m[("India", code.lower())] = "IN-" + code
        for n in names:
            m[("India", fold(n) if not is_native(n) else n)] = "IN-" + code
    for code, names in _FR.items():
        for n in names:
            m[("France", fold(n))] = "FR-" + code
    return m


STATE_MAP = _build_state_map()
# country-agnostic fallback on long unambiguous names / native script
STATE_ANY = {}
for (c, k), v in STATE_MAP.items():
    if len(k) > 3:
        STATE_ANY[k] = v

# A few well known Indian city aliases (generic knowledge, not a lookup service).
CITY_ALIAS = {
    "bombay": "mumbai", "calcutta": "kolkata", "bengaluru": "bangalore", "banglore": "bangalore",
    "gurugram": "gurgaon", "trivandrum": "thiruvananthapuram", "vishakhapatnam": "visakhapatnam",
    "ahmadabad": "ahmedabad", "madras": "chennai", "poona": "pune", "baroda": "vadodara",
    "cochin": "kochi", "mysuru": "mysore", "benares": "varanasi", "kanpur nagar": "kanpur",
}

# ----------------------------------------------------------------------------
# Address token canonicalisation (collapse to short forms)
# ----------------------------------------------------------------------------
ADDR_CANON = {}
for canon, variants in {
    "st": ["street", "str", "st", "saint", "sreet", "stret"],
    "ste": ["sainte"],
    "rd": ["road", "rd"], "ave": ["avenue", "ave", "av", "avn"], "dr": ["drive", "dr", "drv"],
    "ln": ["lane", "ln"], "ct": ["court", "ct", "crt"], "pl": ["place", "pl", "plc"],
    "blvd": ["boulevard", "blvd", "bd", "bvd", "boul"], "pkwy": ["parkway", "pkwy", "pky"],
    "hwy": ["highway", "hwy"], "ter": ["terrace", "ter", "terr"], "cir": ["circle", "cir"],
    "trl": ["trail", "trl"], "sq": ["square", "sq"], "cres": ["crescent", "cres"],
    "mkt": ["market", "mkt"], "ngr": ["nagar", "ngr"], "rue": ["rue", "r"],
    "all": ["allee", "all"], "ch": ["chemin", "ch", "chem"], "imp": ["impasse", "imp"],
    "rte": ["route", "rte", "rt"], "fbg": ["faubourg", "fbg"], "rpt": ["rond-point", "rpt"],
    "qu": ["quai"], "crs": ["cours"], "n": ["north", "n", "nord"], "s": ["south", "s", "sud"],
    "e": ["east", "e", "est"], "w": ["west", "w", "ouest"], "mt": ["mount", "mt"],
    "ft": ["fort", "ft"], "apt": ["apartment", "apt", "flat"], "fl": ["floor", "fl", "flr"],
    "opp": ["opposite", "opp"], "nr": ["near", "nr"], "bldg": ["building", "bldg"],
    "cplx": ["complex", "cmplx"], "soc": ["society", "soc"], "colony": ["colony", "col"],
    "dist": ["district", "dist", "distt"], "tq": ["taluka", "tq", "tal", "taluk"],
    "vill": ["village", "vill", "vpo"], "po": ["post"], "sec": ["sector", "sec"],
    "ph": ["phase", "ph"], "extn": ["extension", "extn", "ext"],
}.items():
    for v in variants:
        ADDR_CANON[v] = canon
ORDINAL_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11",
    "twelfth": "12", "thirteenth": "13", "fourteenth": "14", "fifteenth": "15",
    "sixteenth": "16", "seventeenth": "17", "eighteenth": "18", "nineteenth": "19",
    "twentieth": "20",
}
# tokens carrying no location information
ADDR_DROP = {"no", "number", "unit", "suite", "hno", "h", "door", "plot", "house", "office",
             "shop", "pmb", "box", "po box", "c/o", "co", "the", "of", "de", "du", "des", "la",
             "le", "les", "d", "l", "and", "city", "town", "county", "bis"}

TOKEN_RE = re.compile(r"[a-z0-9]+")
ORD_RE = re.compile(r"^(\d+)(st|nd|rd|th)$")
HN_RE = re.compile(r"^[#\s]*(\d+)(?:-?([a-z]))?(?![a-z0-9])")
HALF_RE = re.compile(r"\b1\s*/\s*2\b")
NUM_RE = re.compile(r"\d+")


def _is_null(c):
    return c.strip().lower() in NULL_TOKENS


def parse_address(addr, country):
    """Return dict with canonical state, house number, number set, tokens, city guess."""
    out = {"state": "", "hn": "", "hn_suf": "", "half": 0, "nums": (), "toks": (), "street": (),
           "city": "", "empty": 1}
    if not isinstance(addr, str) or not addr.strip():
        return out
    comps = [c.strip().strip('"').strip() for c in addr.replace('""', '"').split(",")]
    comps = [c for c in comps if c and not _is_null(c)]
    if not comps:
        return out
    out["empty"] = 0
    # state: the right-most component that is a state name (scanning from the end avoids
    # garbled fragments like "First Flo, Or," being read as a state)
    state = ""
    state_pos = -1
    for j in range(len(comps) - 1, -1, -1):
        c = comps[j]
        key = c if is_native(c) else fold(c).strip(" .")
        st = STATE_MAP.get((country, key)) or (STATE_ANY.get(key) if len(key) > 3 else None)
        if st:
            state, state_pos = st, j
            break
    rest = [c for j, c in enumerate(comps) if j != state_pos]
    out["state"] = state
    folded = [fold(c) for c in rest]
    half = int(any(HALF_RE.search(c) for c in folded))
    folded = [HALF_RE.sub(" ", c) for c in folded]
    # house number = leading number of the first component that starts with a number
    hn = suf = ""
    street_toks = ()
    for c in folded:
        m = HN_RE.match(c)
        if m:
            hn = m.group(1)
            suf = m.group(2) or ""
            street_toks = tuple(_canon_tokens(c[m.end():]))
            break
    if not street_toks:
        # street component without number, e.g. "Mack Rd"
        for c in folded:
            t = _canon_tokens(c)
            if t and t[-1] in ("st", "rd", "ave", "dr", "ln", "ct", "pl", "blvd", "pkwy", "hwy",
                               "ter", "cir", "trl", "rue", "rte", "ch", "imp", "all"):
                street_toks = tuple(t)
                break
    out["hn"] = hn.lstrip("0") or ("0" if hn else "")
    out["hn_suf"] = suf
    out["half"] = half
    out["street"] = street_toks
    nums = set()
    toks = []
    for c in folded:
        for n in NUM_RE.findall(c):
            nums.add(n.lstrip("0") or "0")
        toks.extend(_canon_tokens(c))
    out["nums"] = tuple(sorted(nums))
    out["toks"] = tuple(toks)
    # city guess: last non-state component without digits
    for c in reversed(folded):
        if not any(ch.isdigit() for ch in c):
            cc = c.strip(" .")
            out["city"] = CITY_ALIAS.get(cc, cc)
            break
    return out


def _canon_tokens(s):
    res = []
    for t in TOKEN_RE.findall(s):
        m = ORD_RE.match(t)
        if m:
            res.append(m.group(1))
            continue
        if t in ORDINAL_WORDS:
            res.append(ORDINAL_WORDS[t])
            continue
        if t.isdigit():
            res.append(t.lstrip("0") or "0")
            continue
        t = CITY_ALIAS.get(t, t)
        t = ADDR_CANON.get(t, t)
        if t in ADDR_DROP:
            continue
        res.append(t)
    return res


# ----------------------------------------------------------------------------
# Names
# ----------------------------------------------------------------------------
LEGAL_CANON = {}
for canon, variants in {
    "llc": ["llc", "l l c", "lc"], "inc": ["inc", "incorporated"], "corp": ["corp", "corporation"],
    "co": ["co", "company", "cos"], "ltd": ["ltd", "limited", "ltda"], "pvt": ["pvt", "private", "pte"],
    "lp": ["lp"], "llp": ["llp"], "pllc": ["pllc"], "pc": ["pc", "p c"], "plc": ["plc"],
    "pa": ["pa"], "sarl": ["sarl"], "sas": ["sas"], "sasu": ["sasu"], "sa": ["sa"],
    "eurl": ["eurl"], "sci": ["sci"], "snc": ["snc"], "scop": ["scop"], "gie": ["gie"],
    "opc": ["opc"], "gmbh": ["gmbh"], "ei": ["ei"], "cie": ["cie", "compagnie"],
}.items():
    for v in variants:
        LEGAL_CANON[v] = canon
LEGAL = set(LEGAL_CANON.values())
HONORIFIC = {"dr", "mr", "mrs", "ms", "m/s", "shri", "sri", "shree", "smt", "the", "messrs",
             # French articles / prepositions carry no identity
             "de", "la", "le", "les", "du", "des", "et", "l", "d"}
ALT_RE = re.compile(
    r"\b(?:d\s*/?\s*b\s*/?\s*a|doing business as|formerly known as|formerly|f\s*/?\s*k\s*/?\s*a|"
    r"a\s*/?\s*k\s*/?\s*a|also known as|trading as|t\s*/\s*a)\b\s*:?",
)
DOMAIN_RE = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(?:com|net|org|in|co\.in|fr|co|biz|us|info)\b")
ID_RE = re.compile(r"\(?\b(?:id|ref|reg|no)\s*[:#.]?\s*\d+\)?")
LONGNUM_RE = re.compile(r"[-|]?\s*\b\d{6,}\b")
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "4": "a", "7": "t", "6": "g", "8": "b",
                      "9": "g", "2": "z", "@": "a", "$": "s"})
SINGLE_LETTERS_RE = re.compile(r"\b[a-z](?:\.[a-z])+\b\.?")


MS_RE = re.compile(r"(?<![a-z])m\s*/\s*s(?![a-z])\.?")


def _clean_name(s):
    s = MS_RE.sub(" ", s)
    s = ID_RE.sub(" ", s)
    s = LONGNUM_RE.sub(" ", s)
    s = SINGLE_LETTERS_RE.sub(lambda m: m.group(0).replace(".", "") + " ", s)
    s = s.replace("&", " and ").replace("+", " and ").replace("'", "").replace("`", "")
    return s


def _tokens(s):
    toks = []
    for t in re.findall(r"[a-z0-9]+", s):
        if not t.isdigit() and any(ch.isdigit() for ch in t):
            t = t.translate(LEET)
        toks.append(t)
    return toks


LONG_LEGAL = {"private": "pvt", "limited": "ltd", "corporation": "corp", "incorporated": "inc"}
_NOT_LEGAL = {"primate", "privet", "pirate", "liminal", "limit", "limits", "limitless", "prime",
              "corporate", "corporations", "limbed", "lighted", "privacy", "privat"}
_FUZZY_CACHE = {}


def _fuzzy_legal(t):
    """Misspelled long legal word ("provate", "limted", "lncorporated") -> canonical form."""
    if len(t) < 6 or t.isdigit() or t in _NOT_LEGAL:
        return None
    r = _FUZZY_CACHE.get(t)
    if r is None:
        r = ""
        for w, canon in LONG_LEGAL.items():
            if abs(len(w) - len(t)) > 3:
                continue
            d = Levenshtein.distance(t, w)
            if d <= 1 or (d <= 2 and len(t) >= 7) or (d <= 3 and len(t) >= 9 and
                                                         JaroWinkler.normalized_similarity(t, w) >= 0.80):
                r = canon
                break
        if len(_FUZZY_CACHE) < 200000:
            _FUZZY_CACHE[t] = r
    return r or None


def parse_name(name, translit=None):
    """Return dict: core tokens, legal set, variants, flags.

    translit: optional dict native-token -> latin-token learned from training data.
    """
    out = {"core": (), "legal": (), "alt": (), "native": 0, "domain": 0, "concat": "", "raw": ""}
    if not isinstance(name, str) or not name.strip():
        return out
    s = unicodedata.normalize("NFKC", name)
    if is_native(s):
        out["native"] = 1
        parts = []
        for w in s.split():
            if translit and w in translit:
                parts.append(translit[w])
            else:
                parts.append(unidecode(w))
        s = " ".join(parts)
    s = unidecode(s).lower()
    out["raw"] = s
    dm = DOMAIN_RE.search(s)
    if dm:
        out["domain"] = 1
        s = DOMAIN_RE.sub(lambda m: " " + m.group(1) + " ", s)
    s = _clean_name(s)
    alt = ""
    m = ALT_RE.search(s)
    if m:
        before, after = s[: m.start()], s[m.end():]
        s, alt = after, before
    toks = _tokens(s)
    core, legal = [], []
    i = 0
    while i < len(toks):
        t = toks[i]
        # "l l c", "p c" style legal forms
        if i + 3 <= len(toks) and " ".join(toks[i:i + 3]) in LEGAL_CANON:
            legal.append(LEGAL_CANON[" ".join(toks[i:i + 3])])
            i += 3
            continue
        if t in LEGAL_CANON:
            legal.append(LEGAL_CANON[t])
        elif _fuzzy_legal(t):
            legal.append(_fuzzy_legal(t))
        elif t in HONORIFIC or t == "and":
            pass
        else:
            core.append(t)
        i += 1
    if not core and legal:  # name was only legal words, keep them as core
        core = [t for t in toks if t not in HONORIFIC]
    out["core"] = tuple(core)
    out["legal"] = tuple(sorted(set(legal)))
    if alt:
        out["alt"] = tuple(t for t in _tokens(_clean_name(alt)) if t not in LEGAL_CANON and t not in HONORIFIC)
    out["concat"] = "".join(core)
    return out
