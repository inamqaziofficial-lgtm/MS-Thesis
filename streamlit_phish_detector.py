"""
Streamlit Phishing Detector
Single-file Streamlit app that integrates:
 - URL ML agent (LogisticRegression)
 - Email ML agent (MultinomialNB)
 - Coordinator agent (LogisticRegression)
 - Domain rule-based agent (whois, dns, ssl, lexical heuristics)

How to use:
 - Place the model .pkl files in the same folder as this script (url_agent.pkl, email_agent.pkl, coordinator_agent.pkl,
   url_vectorizer.pkl, email_vectorizer.pkl)
 - Run: `streamlit run streamlit_phish_detector.py`

Notes:
 - If models are not present, the app will still allow domain rule checks but won't run ML predictions.
 - The coordinator was trained expecting a meta input of shape [url_prob, email_prob, is_url, is_email].
 - We provide a slider to blend the URL ML probability with the rule-based score: final_url_prob = blend*ml_prob + (1-blend)*rule_score.

Dependencies:
pip install streamlit joblib whois dnspython tldextract publicsuffix2 requests

"""

import streamlit as st
import joblib, json, socket, ssl, math, tldextract
import whois, dns.resolver, requests
from datetime import datetime

# ------------------------ Utilities ------------------------
def shannon_entropy(s):
    if not s:
        return 0.0
    prob = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum([p * math.log2(p) for p in prob])

def to_naive(dt):
    if isinstance(dt, datetime):
        return dt.replace(tzinfo=None)
    return dt

# ------------------------ Domain rule-based extractor ------------------------
def extract_domain_info(domain):
    info = {"domain": domain}
    now = datetime.utcnow()

    # normalize domain (strip schema/path)
    try:
        # if user passed full url, extract registered domain
        ext = tldextract.extract(domain)
        registered = ext.registered_domain or domain
        info['registered_domain'] = registered
    except Exception:
        info['registered_domain'] = domain

    # --- WHOIS ---
    try:
        w = whois.whois(info['registered_domain'])
        created = w.creation_date
        expires = w.expiration_date
        if isinstance(created, list) and created:
            created = created[0]
        if isinstance(expires, list) and expires:
            expires = expires[0]
        created = to_naive(created)
        expires = to_naive(expires)
        if isinstance(created, datetime):
            info['domain_age_days'] = (now - created).days
        else:
            info['domain_age_days'] = None
        if isinstance(expires, datetime):
            info['days_until_expiry'] = (expires - now).days
        else:
            info['days_until_expiry'] = None
        info['registrar'] = str(w.registrar)
    except Exception as e:
        info['whois_error'] = str(e)
        info['domain_age_days'] = None
        info['days_until_expiry'] = None
        info['registrar'] = None

    # --- DNS A ---
    resolver = dns.resolver.Resolver()
    try:
        a = resolver.resolve(info['registered_domain'], 'A', lifetime=5)
        info['resolved_ips'] = [r.to_text() for r in a]
    except Exception as e:
        info['resolved_ips'] = []
        info['dns_a_error'] = str(e)

    # MX
    try:
        mx = resolver.resolve(info['registered_domain'], 'MX', lifetime=5)
        info['mx_count'] = len(mx)
        info['has_mx'] = True
    except Exception:
        info['mx_count'] = 0
        info['has_mx'] = False

    # NS
    try:
        ns = resolver.resolve(info['registered_domain'], 'NS', lifetime=5)
        info['ns_count'] = len(ns)
    except Exception:
        info['ns_count'] = 0

    # SSL Cert
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=info['registered_domain']) as s:
            s.settimeout(4)
            s.connect((info['registered_domain'], 443))
            cert = s.getpeercert()
            issuer = dict(x[0] for x in cert.get('issuer', ()))
            info['cert_present'] = True
            info['cert_issuer'] = issuer.get('commonName') or issuer.get('CN')
            info['cert_notBefore'] = cert.get('notBefore')
            info['cert_notAfter'] = cert.get('notAfter')
    except Exception as e:
        info['cert_present'] = False
        info['cert_error'] = str(e)

    # HTTP HEAD check
    try:
        r = requests.head(f"https://{info['registered_domain']}", allow_redirects=True, timeout=5)
        info['http_status'] = r.status_code
        info['final_url'] = r.url
    except Exception as e:
        info['http_error'] = str(e)

    # Lexical
    domain_label = (info['registered_domain'] or domain).split(".")[0]
    info['len_domain'] = len(domain_label)
    info['entropy'] = shannon_entropy(domain_label)
    suspicious_tlds = {'.xyz', '.top', '.click', '.tk', '.ml'}
    tld = "." + (info['registered_domain'].split('.')[-1] if '.' in info['registered_domain'] else '')
    info['suspicious_tld'] = tld in suspicious_tlds

    return info

# ------------------------ Simple rule-based scoring ------------------------
def rule_based_score(info):
    """Return a 0..1 score where higher means more suspicious.
    This is a heuristic ensemble of several signals. It's intentionally simple and interpretable.
    """
    score = 0.0
    weight_sum = 0.0

    # short-lived / newly created domains -> suspicious
    if info.get('domain_age_days') is not None:
        w = 2.0
        weight_sum += w
        # <30 days suspicious, >365 days benign
        if info['domain_age_days'] < 30:
            score += w * 1.0
        elif info['domain_age_days'] < 180:
            score += w * 0.6
        else:
            score += w * 0.0

    # about to expire -> suspicious
    if info.get('days_until_expiry') is not None:
        w = 1.5
        weight_sum += w
        if info['days_until_expiry'] < 60:
            score += w * 0.8

    # cert absent
    w = 1.5
    weight_sum += w
    if not info.get('cert_present'):
        score += w * 0.8

    # no A record
    w = 1.0
    weight_sum += w
    if not info.get('resolved_ips'):
        score += w * 0.9

    # suspicious tld
    w = 1.2
    weight_sum += w
    if info.get('suspicious_tld'):
        score += w * 1.0

    # lexical: very high entropy or very long domain
    w = 1.0
    weight_sum += w
    if info.get('entropy', 0) > 3.5:
        score += w * 1.0
    if info.get('len_domain', 0) > 60:
        score += w * 1.0

    # normalize
    if weight_sum == 0:
        return 0.0
    return min(1.0, score / weight_sum)

# ------------------------ Load models ------------------------
@st.cache_resource
def load_models(path_prefix=""):
    models = {}
    names = ["url_agent", "email_agent", "coordinator_agent", "url_vectorizer", "email_vectorizer"]
    for n in names:
        try:
            models[n] = joblib.load(f"{path_prefix}{n}.pkl")
        except Exception as e:
            models[n] = None
    return models

# ------------------------ Prediction helpers ------------------------
def predict_url(url_text, models, blend=0.7):
    # basic normalization
    registered = tldextract.extract(url_text).registered_domain or url_text
    info = extract_domain_info(url_text)
    rule_score = rule_based_score(info)

    url_prob_ml = None
    if models['url_vectorizer'] is not None and models['url_agent'] is not None:
        try:
            v = models['url_vectorizer'].transform([url_text])
            url_prob_ml = float(models['url_agent'].predict_proba(v)[:, 1][0])
        except Exception as e:
            url_prob_ml = None

    if url_prob_ml is None:
        final_url_prob = rule_score
    else:
        final_url_prob = blend * url_prob_ml + (1 - blend) * rule_score

    # coordinator expects [url_prob, email_prob, is_url, is_email]
    meta = [final_url_prob, 0.0, 1.0, 0.0]
    coord_pred = None
    if models['coordinator_agent'] is not None:
        try:
            coord_pred = int(models['coordinator_agent'].predict([meta])[0])
            coord_prob = float(models['coordinator_agent'].predict_proba([meta])[:, 1][0])
        except Exception:
            coord_pred = None
            coord_prob = None
    else:
        coord_prob = None

    return {
        'registered_domain': registered,
        'domain_info': info,
        'rule_score': rule_score,
        'url_ml_prob': url_prob_ml,
        'final_url_prob': final_url_prob,
        'coordinator_pred': coord_pred,
        'coordinator_prob': coord_prob
    }

def predict_email(email_text, models):
    email_prob_ml = None
    if models['email_vectorizer'] is not None and models['email_agent'] is not None:
        try:
            v = models['email_vectorizer'].transform([email_text])
            email_prob_ml = float(models['email_agent'].predict_proba(v)[:, 1][0])
        except Exception:
            email_prob_ml = None

    # meta: [url_prob, email_prob, is_url, is_email]
    meta = [0.0, email_prob_ml or 0.0, 0.0, 1.0]
    coord_pred = None
    coord_prob = None
    if models['coordinator_agent'] is not None:
        try:
            coord_pred = int(models['coordinator_agent'].predict([meta])[0])
            coord_prob = float(models['coordinator_agent'].predict_proba([meta])[:, 1][0])
        except Exception:
            coord_pred = None

    return {
        'email_ml_prob': email_prob_ml,
        'coordinator_pred': coord_pred,
        'coordinator_prob': coord_prob
    }

# ------------------------ Streamlit UI ------------------------
st.set_page_config(page_title="Phishing Detector", layout='wide')
st.title("🛡️ Phishing Detection — Streamlit Interface")
st.write("Combine rule-based domain checks with trained ML agents (URL, Email) and a coordinator meta-model.")

# model path prefix
st.sidebar.header("Settings")
path_prefix = st.sidebar.text_input("Model path prefix (folder/ or leave blank)", value="")
models = load_models(path_prefix)

st.sidebar.markdown("**Loaded models**")
for k,v in models.items():
    st.sidebar.write(f"{k}: {'✅' if v is not None else '❌ not found'}")

mode = st.radio("Choose input type:", ["URL", "Email"], index=0, horizontal=True)

if mode == "URL":
    col1, col2 = st.columns([2,1])
    with col1:
        url_in = st.text_input("Enter URL or domain", value="http://example.com")
        blend = st.slider("Blend ML URL prob with rule-based score (ML weight)", 0.0, 1.0, 0.7)
        if st.button("Analyze URL"):
            with st.spinner("Running checks — may take a few seconds for whois/dns/ssl..."):
                res = predict_url(url_in, models, blend=blend)
            st.subheader("Summary")
            st.metric("Final suspiciousness (0-1)", f"{res['final_url_prob']:.3f}")
            if res['url_ml_prob'] is not None:
                st.write(f"URL ML probability: {res['url_ml_prob']:.3f}")
            st.write(f"Rule-based score: {res['rule_score']:.3f}")
            if res['coordinator_pred'] is not None:
                st.write(f"Coordinator prediction: {'PHISH' if res['coordinator_pred']==1 else 'BENIGN'} (prob {res['coordinator_prob']:.3f})")

            with st.expander("Domain & DNS details (JSON)"):
                st.json(res['domain_info'])

            with st.expander("Explainability — how rule score computed"):
                st.write("The rule-based score uses domain age, expiry, SSL presence, DNS records, TLD reputation, and lexical heuristics. "
                         "You can adjust the blend slider to combine ML and rule-based signals.")

else:  # Email
    col1, col2 = st.columns([2,1])
    with col1:
        email_in = st.text_area("Paste full email text (body + subject) below")
        if st.button("Analyze Email"):
            with st.spinner("Vectorizing and predicting..."):
                res = predict_email(email_in, models)
            st.subheader("Summary")
            if res['email_ml_prob'] is not None:
                st.metric("Email phishing probability (ML)", f"{res['email_ml_prob']:.3f}")
            else:
                st.write("Email ML model not available.")
            if res['coordinator_pred'] is not None:
                st.write(f"Coordinator prediction: {'PHISH' if res['coordinator_pred']==1 else 'BENIGN'} (prob {res['coordinator_prob']:.3f})")

            with st.expander("Raw outputs (JSON)"):
                st.json(res)

# Footer / help
st.markdown("---")
st.write("**Notes:**\n- If models are missing, download or paste trained .pkl files into the app folder.\n- Running whois/dns/ssl checks requires outbound network access and may be slow or blocked depending on environment.")

st.write("If you want, I can help produce a version that trains models from CSV files directly in the app — say so and I will add a 'Train models' section (warning: training in Streamlit can be slow).")
