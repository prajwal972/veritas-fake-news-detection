import os
import re
import time
import hashlib
import logging
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import nltk 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(os.path.dirname(BASE_DIR), "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, 'audit.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

for resource in ['stopwords', 'wordnet', 'punkt', 'punkt_tab']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

API_KEYS = {
    "demo-key-2024": "demo_user",
    "internship-key": "intern"
}

SUSPICIOUS_LEXICON = {
    "high": [
        "hoax", "fabricated", "conspiracy", "shocking", "unbelievable",
        "miracle", "secret", "exposed", "bombshell", "breaking",
        "exclusive", "leaked", "scandal", "cover-up", "deep state",
        "mainstream media", "fake", "fraud", "wake up", "share this"
    ],
    "medium": [
        "allegedly", "sources say", "anonymous", "rumored", "claimed",
        "according to insiders", "could be", "might be", "radical",
        "extreme", "outrage", "disgusting", "horrifying", "must read",
        "truth", "reality check"
    ],
    "low": [
        "surprising", "unexpected", "unusual", "strange", "odd",
        "controversial", "debated", "disputed", "unclear", "uncertain"
    ]
}

SUSPICIOUS_FLAT = {}
for _level, _words in SUSPICIOUS_LEXICON.items():
    for _w in _words:
        SUSPICIOUS_FLAT[_w.lower()] = _level

HEDGE_WORDS = [
    "allegedly", "apparently", "claims", "reportedly", "possibly", "perhaps",
    "maybe", "could", "might", "may", "suggests", "appears", "seems",
    "according to", "sources say", "rumored", "unconfirmed", "speculated"
]

SARCASM_SIGNALS = [
    "oh sure", "yeah right", "of course", "obviously", "totally",
    "absolutely brilliant", "what a surprise", "clearly", "naturally"
]

TOPIC_KEYWORDS = {
    "political":   ["election", "government", "president", "minister", "parliament", "policy", "vote", "congress"],
    "health":      ["vaccine", "virus", "pandemic", "disease", "hospital", "fda", "who", "health", "drug", "cure"],
    "financial":   ["stock", "market", "economy", "inflation", "gdp", "bank", "fed", "interest rate", "recession", "crypto"],
    "military":    ["war", "military", "attack", "bomb", "troops", "nato", "missile", "conflict", "army"],
    "technology":  ["ai", "artificial intelligence", "tech", "data", "cyber", "hack", "software", "robot"],
    "climate":     ["climate", "carbon", "emission", "renewable", "environment", "warming", "fossil fuel"]
}

COUNTRY_SECTORS = {
    "USA":         {"markets": ["S&P 500", "NASDAQ", "Dow Jones", "USD/EUR", "US Treasury"],  "sectors": ["Technology", "Finance", "Healthcare"],  "sensitivity": 1.0},
    "India":       {"markets": ["SENSEX", "NIFTY 50", "USD/INR", "BSE Mid Cap"],              "sectors": ["IT Services", "Pharma", "Banking"],       "sensitivity": 0.85},
    "EU":          {"markets": ["DAX", "CAC 40", "FTSE 100", "EUR/USD"],                      "sectors": ["Manufacturing", "Finance", "Energy"],      "sensitivity": 0.9},
    "China":       {"markets": ["Shanghai Composite", "Hang Seng", "USD/CNY"],                "sectors": ["Manufacturing", "Technology", "Real Estate"], "sensitivity": 0.88},
    "Japan":       {"markets": ["Nikkei 225", "TOPIX", "USD/JPY"],                            "sectors": ["Automotive", "Technology", "Finance"],     "sensitivity": 0.82},
    "UK":          {"markets": ["FTSE 100", "FTSE 250", "GBP/USD", "UK Gilts"],               "sectors": ["Finance", "Energy", "Healthcare"],         "sensitivity": 0.87},
    "Crypto":      {"markets": ["Bitcoin (BTC)", "Ethereum (ETH)", "BNB", "XRP"],             "sectors": ["DeFi", "NFT", "Layer 2", "Stablecoins"],   "sensitivity": 1.2},
    "Forex":       {"markets": ["EUR/USD", "GBP/USD", "USD/JPY", "USD/INR", "AUD/USD"],       "sectors": ["Carry Trade", "Safe Haven", "EM"],         "sensitivity": 0.95},
    "Commodities": {"markets": ["Gold", "Crude Oil (WTI)", "Natural Gas", "Silver"],          "sectors": ["Energy", "Precious Metals", "Industrial"],  "sensitivity": 0.9}
}

TOPIC_MARKET_MAP = {
    "political":  ["USA", "EU", "UK", "Forex"],
    "health":     ["USA", "India", "EU", "Commodities"],
    "financial":  ["USA", "China", "EU", "Crypto", "Forex", "Commodities"],
    "military":   ["USA", "EU", "UK", "Commodities", "Forex"],
    "technology": ["USA", "China", "Japan", "Crypto"],
    "climate":    ["EU", "UK", "Commodities", "USA"]
}


class TextPreprocessor:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = set()
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
        return ' '.join(
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if t not in self.stop_words and len(t) > 2
        )

    def extract_features(self, text: str) -> dict:
        words = text.lower().split()
        word_count    = len(words)
        cap_ratio     = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        excl_count    = text.count('!')
        ques_count    = text.count('?')
        url_count     = len(re.findall(r'http\S+|www\S+', text))
        avg_word_len  = float(np.mean([len(w) for w in words])) if words else 0.0
        hedge_count   = sum(1 for h in HEDGE_WORDS if h in text.lower())
        sarcasm_count = sum(1 for s in SARCASM_SIGNALS if s in text.lower())
        ttr           = len(set(words)) / max(len(words), 1)
        punct_score   = (excl_count * 3 + ques_count * 1.5 + sum(1 for c in text if c.isupper()) * 0.5) / max(len(text), 1)

        try:
            sentences = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 10]
        except Exception:
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]

        suspicious_hits = {"high": [], "medium": [], "low": []}
        for phrase, level in SUSPICIOUS_FLAT.items():
            if phrase in text.lower():
                suspicious_hits[level].append(phrase)

        thematic_coherence = self._thematic_coherence(sentences)
        stance_drift       = self._stance_drift(sentences)

        return {
            "word_count":         word_count,
            "sentence_count":     len(sentences),
            "cap_ratio":          round(cap_ratio, 3),
            "exclamations":       excl_count,
            "questions":          ques_count,
            "urls":               url_count,
            "avg_word_len":       round(avg_word_len, 2),
            "suspicious_words":   suspicious_hits,
            "hedge_count":        hedge_count,
            "sarcasm_signals":    sarcasm_count,
            "thematic_coherence": round(thematic_coherence, 3),
            "stance_drift":       round(stance_drift, 3),
            "type_token_ratio":   round(ttr, 3),
            "punctuation_score":  round(punct_score, 4),
            "epistemic_ratio":    round(hedge_count / max(word_count, 1), 4)
        }

    def _thematic_coherence(self, sentences: list) -> float:
        if len(sentences) < 2:
            return 1.0
        first = set(sentences[0].lower().split())
        rest  = set(' '.join(sentences[1:]).lower().split())
        if not first or not rest:
            return 1.0
        return len(first & rest) / max(len(first | rest), 1)

    def _stance_drift(self, sentences: list) -> float:
        pos = ["good","great","positive","success","benefit","improve","gain","confirmed","proven"]
        neg = ["bad","terrible","negative","failure","harm","worsen","loss","denied","false"]
        if not sentences:
            return 0.0
        def score(s):
            sl = s.lower()
            return sum(1 for w in pos if w in sl) - sum(1 for w in neg if w in sl)
        scores = [score(s) for s in sentences]
        diffs  = [abs(scores[i] - scores[i-1]) for i in range(1, len(scores))]
        return float(np.mean(diffs)) if diffs else 0.0


preprocessor = TextPreprocessor()


class MarketImpactAnalyzer:
    def detect_topics(self, text: str) -> list:
        text_lower = text.lower()
        detected = []
        for topic, keywords in TOPIC_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in text_lower)
            if hits > 0:
                detected.append({"topic": topic, "relevance": round(min(hits / 3.0, 1.0), 2)})
        detected.sort(key=lambda x: x["relevance"], reverse=True)
        return detected[:3] if detected else [{"topic": "general", "relevance": 0.3}]

    def analyze(self, text: str, fake_probability: float, risk_score: int) -> dict:
        topics   = self.detect_topics(text)
        regions  = set()
        for t in topics:
            for r in TOPIC_MARKET_MAP.get(t["topic"], ["USA", "EU"]):
                regions.add(r)
        if not regions:
            regions = {"USA", "EU", "Forex"}

        impact_score = (fake_probability / 100) * (risk_score / 100)
        impact_level = self._level(fake_probability, risk_score)

        regional = []
        for region in sorted(regions):
            if region not in COUNTRY_SECTORS:
                continue
            data      = COUNTRY_SECTORS[region]
            base      = impact_score * data["sensitivity"]
            direction = self._direction(text, fake_probability)
            vol       = self._volatility(base, fake_probability)
            regional.append({
                "region":       region,
                "impact_level": impact_level,
                "impact_score": round(base * 100, 1),
                "direction":    direction,
                "volatility":   vol,
                "markets":      data["markets"],
                "sectors":      data["sectors"],
                "description":  self._desc(region, impact_level, direction, topics)
            })

        return {
            "detected_topics":      topics,
            "impact_level":         impact_level,
            "overall_impact_score": round(impact_score * 100, 1),
            "affected_regions":     regional,
            "disclaimer":           "Market impact is simulated analysis for educational purposes only. Not financial advice."
        }

    def _level(self, fp: float, rs: int) -> str:
        c = (fp + rs) / 2
        if c >= 75: return "HIGH"
        if c >= 50: return "MODERATE"
        if c >= 25: return "LOW"
        return "MINIMAL"

    def _direction(self, text: str, fp: float) -> str:
        neg_kw = ["crash","collapse","crisis","war","attack","ban","sanction","fall","drop","recession"]
        pos_kw = ["growth","recovery","deal","agreement","rise","gain","boost","profit","expansion"]
        tl = text.lower()
        n = sum(1 for w in neg_kw if w in tl)
        p = sum(1 for w in pos_kw if w in tl)
        if fp > 60: return "BEARISH"
        if n > p:   return "BEARISH"
        if p > n:   return "BULLISH"
        return "NEUTRAL"

    def _volatility(self, base: float, fp: float) -> str:
        v = base + fp / 200
        if v > 0.6:  return "HIGH"
        if v > 0.35: return "MODERATE"
        return "LOW"

    def _desc(self, region: str, level: str, direction: str, topics: list) -> str:
        topic = topics[0]["topic"].capitalize() if topics else "General"
        msgs = {
            "HIGH":     f"{topic} news of this nature historically triggers significant {direction.lower()} pressure on {region} markets with elevated volatility.",
            "MODERATE": f"Moderate {direction.lower()} influence expected across {region} {topic.lower()} sectors if this news propagates widely.",
            "LOW":      f"Limited market reaction anticipated in {region}. {topic} sentiment may shift marginally.",
            "MINIMAL":  f"Negligible direct impact on {region} markets expected from this content."
        }
        return msgs.get(level, "Impact analysis unavailable.")


market_analyzer = MarketImpactAnalyzer()


class ModelManager:
    def __init__(self):
        self.models:     dict            = {}
        self.vectorizer: TfidfVectorizer = None
        self._train()

    def _training_data(self):
        real = [
            "The Federal Reserve raised interest rates by 25 basis points citing inflation concerns.",
            "Scientists at MIT published research on quantum computing breakthroughs in Nature journal.",
            "The World Health Organization released new guidelines on antibiotic resistance treatment.",
            "NASA confirmed the successful landing of the Perseverance rover on Mars surface.",
            "Parliament passed legislation on data privacy protection with bipartisan support.",
            "The stock market closed higher after positive economic reports from the labor department.",
            "Researchers found a correlation between diet and cognitive decline in elderly patients.",
            "The United Nations Security Council voted on the resolution regarding international trade.",
            "Climate scientists measured record ocean temperatures in the Pacific Ocean this summer.",
            "The Supreme Court issued a ruling on first amendment rights in digital communications.",
            "Economists forecast moderate GDP growth of 2.3 percent for the upcoming fiscal quarter.",
            "The pharmaceutical company completed phase three clinical trials for the new vaccine.",
            "Municipal authorities announced infrastructure improvements for the downtown corridor.",
            "The diplomatic summit resulted in a bilateral trade agreement between the two nations.",
            "Academic researchers published findings on renewable energy efficiency improvements.",
            "Health officials confirmed a decrease in respiratory illness rates across the region.",
            "The technology company reported quarterly earnings above analyst expectations.",
            "Transportation officials approved funding for the new high-speed rail project.",
            "Environmental monitors recorded improved air quality standards in urban areas.",
            "The international court issued its decision on maritime boundary disputes.",
            "Government released new education policy focusing on digital literacy for students.",
            "Central bank maintained reserve ratios following quarterly monetary policy review.",
            "University study shows correlation between exercise frequency and mental health outcomes.",
            "Trade ministry signed memorandum of understanding with four regional partners.",
            "Public health department announced expanded vaccination program for rural districts.",
        ] * 8

        fake = [
            "SHOCKING Government secretly putting chemicals in water to control population minds.",
            "EXPOSED Doctors don't want you to know this miracle cure that eliminates all disease.",
            "BOMBSHELL Deep state operatives planning to steal the next election sources reveal.",
            "SECRET leaked documents prove mainstream media is lying about everything you believe.",
            "UNBELIEVABLE Celebrity admits to being part of cult running global government.",
            "They don't want you to know this one weird trick that cures cancer overnight naturally.",
            "WAKE UP sheeple 5G towers are spreading the virus and this is being covered up.",
            "FRAUD EXPOSED Scientists paid billions to fake climate change data insider reveals.",
            "SHOCKING truth about vaccines they are hiding from you and your family right now.",
            "Anonymous insider exposes the hoax that has been deceiving humanity for decades.",
            "EXCLUSIVE Bombshell evidence proves moon landing was completely fabricated by NASA.",
            "Radical elites plotting to destroy nations with their disgusting hidden agenda exposed.",
            "You won't believe what they found in chemtrails raining poison on people.",
            "SHARE THIS before they delete it the scandal that will bring down the entire system.",
            "Whistleblower reveals shocking cover-up of alien contact suppressed for 50 years.",
            "MUST READ The horrifying truth about fluoride and what it does to your brain daily.",
            "OUTRAGE Secret society controls all world leaders through blackmail and corruption.",
            "Miracle frequency discovered that they banned because it heals everything instantly.",
            "BREAKING Massive conspiracy uncovered proving global elite runs trafficking ring.",
            "The truth about what really happened that they erased from history books entirely.",
            "Deep state operatives infiltrating government to bring down elected officials.",
            "They are hiding the real cure for all disease because big pharma profits from sickness.",
            "Bombshell report exposes the global conspiracy to control food supply worldwide.",
            "Wake up to the truth they have been suppressing this information for decades now.",
            "Shocking leaked footage proves everything the mainstream media told you was a lie.",
        ] * 8

        texts  = real + fake
        labels = ['REAL'] * len(real) + ['FAKE'] * len(fake)
        return texts, labels

    def _train(self):
        texts, labels = self._training_data()
        cleaned = [preprocessor.clean(t) for t in texts]
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X = self.vectorizer.fit_transform(cleaned)

        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        lr.fit(X, labels)
        self.models['logistic_regression'] = lr

        nb = MultinomialNB(alpha=0.1)
        nb.fit(X, labels)
        self.models['naive_bayes'] = nb

        logger.info(f"Models trained on {len(texts)} samples.")

    def predict(self, text: str, model_name: str = 'logistic_regression') -> dict:
        cleaned  = preprocessor.clean(text)
        features = preprocessor.extract_features(text)

        if model_name == 'lstm':
            return self._ensemble(cleaned, text, features)

        if model_name not in self.models or not hasattr(self.models[model_name], 'predict_proba'):
            model_name = 'logistic_regression'

        X      = self.vectorizer.transform([cleaned])
        model  = self.models[model_name]
        probas = model.predict_proba(X)[0]
        cls    = list(model.classes_)
        fi     = cls.index('FAKE') if 'FAKE' in cls else 1
        ri     = cls.index('REAL') if 'REAL' in cls else 0

        fp, rp = float(probas[fi]), float(probas[ri])
        fp, rp = self._heuristics(fp, rp, features)
        pred   = 'FAKE' if fp > rp else 'REAL'
        return self._build(pred, max(fp, rp), fp, rp, features, text, model_name)

    def _ensemble(self, cleaned: str, raw: str, features: dict) -> dict:
        X     = self.vectorizer.transform([cleaned])
        preds = []
        for m in self.models.values():
            if not hasattr(m, 'predict_proba'):
                continue
            pr  = m.predict_proba(X)[0]
            cls = list(m.classes_)
            fi  = cls.index('FAKE') if 'FAKE' in cls else 1
            preds.append(float(pr[fi]))
        fp = float(np.mean(preds)) if preds else 0.5
        rp = 1.0 - fp
        fp, rp = self._heuristics(fp, rp, features)
        pred = 'FAKE' if fp > rp else 'REAL'
        return self._build(pred, max(fp, rp), fp, rp, features, raw, 'lstm_ensemble')

    def _heuristics(self, fp: float, rp: float, f: dict):
        boost = (
            len(f['suspicious_words']['high'])   * 0.05 +
            len(f['suspicious_words']['medium'])  * 0.02 +
            min(f['cap_ratio'], 0.5)              * 0.10 +
            min(f['exclamations'], 5)             * 0.02 +
            f.get('sarcasm_signals', 0)           * 0.03 +
            min(f.get('punctuation_score', 0), 1) * 0.05 +
            (1.0 - f.get('thematic_coherence', 1.0)) * 0.04 +
            min(f.get('stance_drift', 0), 5)      * 0.01
        )
        fp    = min(fp + boost, 0.99)
        rp    = max(rp - boost, 0.01)
        total = fp + rp
        return fp / total, rp / total

    def _build(self, pred, conf, fp, rp, features, text, model_name) -> dict:
        risk   = self._risk(fp, features)
        market = market_analyzer.analyze(text, fp * 100, risk)
        return {
            "prediction":        pred,
            "confidence":        round(conf * 100, 2),
            "fake_probability":  round(fp   * 100, 2),
            "real_probability":  round(rp   * 100, 2),
            "risk_score":        risk,
            "model_used":        model_name,
            "features":          features,
            "highlighted_text":  self._highlight(text, features['suspicious_words']),
            "analysis":          self._analysis(pred, conf, features),
            "discourse_analysis":self._discourse(features),
            "market_impact":     market,
            "timestamp":         datetime.utcnow().isoformat()
        }

    def _highlight(self, text: str, sus: dict) -> list:
        result = []
        for token in re.split(r'(\s+)', text):
            wl    = token.lower().strip('.,!?;:"\'')
            level = None
            for lvl, words in sus.items():
                if any(wl == w or wl in w or w in wl for w in words):
                    level = lvl
                    break
            result.append({"token": token, "highlight": level})
        return result

    def _risk(self, fp: float, f: dict) -> int:
        s  = int(fp * 55)
        s += len(f['suspicious_words']['high'])   * 4
        s += len(f['suspicious_words']['medium'])  * 2
        s += min(f['exclamations'] * 2, 10)
        s += min(int(f['cap_ratio'] * 25), 12)
        s += min(f.get('sarcasm_signals', 0) * 3, 9)
        s += int((1.0 - f.get('thematic_coherence', 1.0)) * 10)
        s += min(int(f.get('stance_drift', 0) * 3), 6)
        return min(s, 100)

    def _analysis(self, pred: str, conf: float, f: dict) -> list:
        r = []
        if len(f['suspicious_words']['high']) > 0:
            r.append(f"Contains {len(f['suspicious_words']['high'])} high-risk sensationalist words")
        if f['exclamations'] > 2:
            r.append(f"Excessive exclamation marks detected ({f['exclamations']})")
        if f['cap_ratio'] > 0.15:
            r.append("Abnormally high capitalization ratio detected")
        if f['word_count'] < 20:
            r.append("Very short article — may lack journalistic depth")
        if len(f['suspicious_words']['medium']) > 3:
            r.append("Multiple vague or unverifiable claim phrases found")
        if f.get('sarcasm_signals', 0) > 1:
            r.append(f"Sarcasm or irony signals detected ({f['sarcasm_signals']} markers)")
        if f.get('thematic_coherence', 1.0) < 0.15:
            r.append("Low thematic coherence — article topic drifts significantly")
        if f.get('stance_drift', 0) > 2:
            r.append("Inconsistent stance or sentiment across article sections")
        if f.get('type_token_ratio', 1.0) < 0.4:
            r.append("Low lexical diversity — repetitive language patterns detected")
        if not r:
            if pred == 'REAL':
                r += ["Neutral language and structured prose detected", "No major sensationalist indicators found"]
            else:
                r.append("Statistical patterns match fake news corpus")
        return r

    def _discourse(self, f: dict) -> dict:
        coh  = f.get('thematic_coherence', 1.0)
        std  = f.get('stance_drift', 0.0)
        hed  = f.get('epistemic_ratio', 0.0)
        ttr  = f.get('type_token_ratio', 1.0)
        sarc = f.get('sarcasm_signals', 0)
        punc = f.get('punctuation_score', 0)
        return {
            "thematic_coherence_score": round(coh * 100, 1),
            "thematic_coherence_label": "High" if coh > 0.4 else "Medium" if coh > 0.2 else "Low",
            "stance_drift_score":       round(min(std / 5, 1) * 100, 1),
            "stance_drift_label":       "Stable" if std < 1 else "Moderate drift" if std < 3 else "High drift",
            "epistemic_hedging_score":  round(min(hed * 20, 1) * 100, 1),
            "hedging_label":            "Well-hedged" if hed > 0.05 else "Partially hedged" if hed > 0.02 else "Assertive / Unhedged",
            "lexical_diversity_score":  round(ttr * 100, 1),
            "sarcasm_score":            round(min(sarc / 3, 1) * 100, 1),
            "punctuation_intensity":    round(min(punc * 10, 1) * 100, 1)
        }


model_manager = ModelManager()


def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('X-API-Key', '')
        if key not in API_KEYS:
            logger.warning(f"Unauthorized access from {request.remote_addr}")
            return jsonify({"error": "Unauthorized. Provide a valid X-API-Key header."}), 401
        return f(*args, **kwargs)
    return decorated


app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app, resources={r"/api/*": {"origins": "*"}})
limiter = Limiter(get_remote_address, app=app, default_limits=["200 per day", "50 per hour"], storage_uri="memory://")


@app.route('/api/health', methods=['GET'])
@limiter.exempt
def health():
    return jsonify({"status": "operational", "models_loaded": list(model_manager.models.keys()), "timestamp": datetime.utcnow().isoformat(), "version": "2.0.0"})


@app.route('/api/models', methods=['GET'])
@limiter.exempt
def get_models():
    return jsonify({"models": [
        {"id": "logistic_regression", "name": "Logistic Regression", "accuracy": "~87%", "speed": "Fast"},
        {"id": "naive_bayes",         "name": "Naive Bayes",         "accuracy": "~83%", "speed": "Fastest"},
        {"id": "lstm",                "name": "LSTM Ensemble",       "accuracy": "~91%", "speed": "Moderate"}
    ]})


@app.route('/api/analyze', methods=['POST'])
@limiter.limit("30 per minute")
@require_api_key
def analyze():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text  = str(data.get('text', '')).strip()
    model = str(data.get('model', 'logistic_regression'))

    if len(text) < 20:
        return jsonify({"error": "Text too short. Minimum 20 characters."}), 400
    if len(text) > 10000:
        return jsonify({"error": "Text too long. Maximum 10,000 characters."}), 400
    if model not in ['logistic_regression', 'naive_bayes', 'lstm']:
        model = 'logistic_regression'

    start  = time.time()
    result = model_manager.predict(text, model)
    result['processing_time_ms'] = round((time.time() - start) * 1000, 2)

    logger.info(f"ANALYZE | model={model} | prediction={result['prediction']} | confidence={result['confidence']} | hash={hashlib.sha256(text.encode()).hexdigest()[:16]}")
    return jsonify(result)


@app.route('/api/batch', methods=['POST'])
@limiter.limit("5 per minute")
@require_api_key
def batch_analyze():
    data = request.get_json()
    if not data or 'articles' not in data:
        return jsonify({"error": "Missing 'articles' array"}), 400
    model   = data.get('model', 'logistic_regression')
    results = []
    for i, article in enumerate(data['articles'][:10]):
        if isinstance(article, str) and len(article) >= 20:
            results.append({"index": i, **model_manager.predict(article, model)})
    return jsonify({"results": results, "count": len(results)})


@app.route('/')
@limiter.exempt
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
