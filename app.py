"""
Makroinstrument: amerikanska räntor, Dollar Index, USD/JPY och guld.
Starta:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Makroinstrument", layout="centered",
                   initial_sidebar_state="collapsed")

TICKERS = {
    "US 3M": "^IRX",
    "US 5Y": "^FVX",
    "US 10Y": "^TNX",
    "US 30Y": "^TYX",
    "Dollar Index": "DX-Y.NYB",
    "Guld": "GC=F",
    "USD/JPY": "JPY=X",
}
YIELDS = ["US 3M", "US 5Y", "US 10Y", "US 30Y"]
TIMEFRAMES = {
    "Dag (2 år)": ("2y", "1d"),
    "Timme (60 dagar)": ("60d", "1h"),
    "5 min (30 dagar)": ("30d", "5m"),
}


@st.cache_data(ttl=60, show_spinner="Hämtar data...")
def load(period: str, interval: str) -> pd.DataFrame:
    raw = yf.download(
        list(TICKERS.values()), period=period, interval=interval,
        auto_adjust=True, progress=False,
    )["Close"]
    raw = raw.rename(columns={v: k for k, v in TICKERS.items()})
    return raw.ffill().dropna(how="all")


def moves(df: pd.DataFrame) -> pd.DataFrame:
    """Räntor: absolut förändring (procentenheter). Övrigt: procentuell förändring."""
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        out[c] = df[c].diff() if c in YIELDS else df[c].pct_change()
    return out.dropna(how="all")


# ---------------- Inställningar (på sidan, mobilvänligt) ----------------
st.title("Makroinstrument")
with st.expander("Inställningar", expanded=False):
    tf_name = st.selectbox("Tidsram", list(TIMEFRAMES))
    window = st.slider("Rullande fönster (staplar)", 10, 200, 40)
    if st.button("Uppdatera nu"):
        st.cache_data.clear()
period, interval = TIMEFRAMES[tf_name]

df = load(period, interval)
if df.empty:
    st.error("Ingen data hämtades. Kontrollera internetanslutningen.")
    st.stop()
mv = moves(df)

st.caption(f"Senaste datapunkt: {df.index[-1]}  |  Källa: Yahoo Finance (kan vara fördröjd)")

# ---------------- 1. Nuläge ----------------
st.subheader("1. Nuläge")
names = list(df.columns)
for i in range(0, len(names), 2):
    cols = st.columns(2)
    for col, name in zip(cols, names[i:i + 2]):
        s_ = df[name].dropna()
        if len(s_) < 2:
            continue
        last, prev = s_.iloc[-1], s_.iloc[-2]
        if name in YIELDS:
            col.metric(name, f"{last:.3f} %", f"{(last - prev) * 100:+.1f} bp")
        else:
            col.metric(name, f"{last:,.2f}", f"{(last / prev - 1) * 100:+.2f} %")

# ---------------- 2. Räntekurva ----------------
st.subheader("2. Räntekurva")
curve = pd.DataFrame({
    "10Y − 3M": df["US 10Y"] - df["US 3M"],
    "30Y − 5Y": df["US 30Y"] - df["US 5Y"],
}).dropna()
fig = go.Figure()
for c in curve.columns:
    fig.add_trace(go.Scatter(x=curve.index, y=curve[c], name=c))
fig.add_hline(y=0, line_dash="dot")
fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation='h', y=-0.2), yaxis_title="Procentenheter")
st.plotly_chart(fig, use_container_width=True)
if not curve.empty:
    s10 = curve["10Y − 3M"].iloc[-1]
    st.write(f"**10Y − 3M:** {s10:+.2f} pe")
    st.write("Inverterad kurva (negativ) har historiskt föregått lågkonjunkturer."
             if s10 < 0 else "Normal kurva (positiv).")

# ---------------- 3. Normaliserad jämförelse ----------------
st.subheader("3. Jämförelse (indexerad till 100 vid start)")
pick = st.multiselect("Visa", ["Dollar Index", "Guld", "USD/JPY", "US 10Y"],
                      default=["Dollar Index", "Guld", "USD/JPY"])
if pick:
    norm = df[pick].dropna()
    norm = norm / norm.iloc[0] * 100
    fig = go.Figure([go.Scatter(x=norm.index, y=norm[c], name=c) for c in norm.columns])
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation='h', y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

# ---------------- 4. Samband ----------------
st.subheader("4. Samband (korrelation mellan rörelser)")
st.caption("−1 = rör sig motsatt, 0 = inget samband, +1 = rör sig likadant.")
PAIRS = [("Guld", "Dollar Index"), ("Guld", "US 10Y"),
         ("USD/JPY", "US 10Y"), ("USD/JPY", "Dollar Index")]
rows = []
for a, b in PAIRS:
    pair = mv[[a, b]].dropna()
    full = pair[a].corr(pair[b])
    recent = pair.tail(window)[a].corr(pair.tail(window)[b])
    rows.append({"Par": f"{a} vs {b}", "Hela perioden": round(full, 2),
                 f"Senaste {window}": round(recent, 2)})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

a_sel = st.selectbox("Rullande korrelation för", [f"{a} vs {b}" for a, b in PAIRS])
a, b = a_sel.split(" vs ")
roll = mv[a].rolling(window).corr(mv[b]).dropna()
fig = go.Figure(go.Scatter(x=roll.index, y=roll, name=a_sel))
fig.add_hline(y=0, line_dash="dot")
fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), yaxis_range=[-1, 1])
st.plotly_chart(fig, use_container_width=True)

# ---------------- 5. Tidsfördröjning ----------------
st.subheader("5. Går något före? (test av 'köp innan det händer')")
st.caption(
    "Korrelerar rörelsen i A nu med rörelsen i B för k staplar sedan. "
    "Om k > 0 har ett tydligt utslag betyder det att B leder A. "
    "Staplar utanför de röda linjerna är statistiskt skilda från noll."
)
cA, cB = st.columns(2)
A = cA.selectbox("A (det som följer efter)", df.columns, index=list(df.columns).index("Guld"))
B = cB.selectbox("B (möjlig ledare)", df.columns, index=list(df.columns).index("Dollar Index"))
if A != B:
    pair = mv[[A, B]].dropna()
    lags = list(range(-5, 6))
    vals = [pair[A].corr(pair[B].shift(k)) for k in lags]
    n = len(pair)
    thr = 2 / np.sqrt(n)
    fig = go.Figure(go.Bar(x=lags, y=vals))
    fig.add_hline(y=thr, line_color="red", line_dash="dash")
    fig.add_hline(y=-thr, line_color="red", line_dash="dash")
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="Fördröjning k (staplar)", yaxis_title="Korrelation")
    st.plotly_chart(fig, use_container_width=True)
    best = lags[int(np.nanargmax(np.abs(vals)))]
    if best == 0:
        st.info("Starkast samband vid k = 0: rörelserna sker samtidigt. "
                "Ingen användbar fördröjning att handla på.")
    elif abs(vals[lags.index(best)]) > thr:
        st.warning(f"Utslag vid k = {best}. Kontrollera att det håller på andra perioder "
                   "och efter courtage/spread innan du tror på det.")
    else:
        st.info("Inga fördröjningar är statistiskt signifikanta.")

# ---------------- 6. Lägesbild ----------------
st.subheader("6. Lägesbild (beskrivande, ingen rekommendation)")
look = min(window, len(df) - 1)


def chg(name: str) -> float:
    s = df[name].dropna()
    return (s.iloc[-1] - s.iloc[-look - 1]) if name in YIELDS else (s.iloc[-1] / s.iloc[-look - 1] - 1) * 100


notes = []
d10, dxy, gold, jpy = chg("US 10Y"), chg("Dollar Index"), chg("Guld"), chg("USD/JPY")
notes.append(f"Över senaste {look} staplarna: 10Y {d10:+.2f} pe, DXY {dxy:+.2f} %, "
             f"guld {gold:+.2f} %, USD/JPY {jpy:+.2f} %.")
if dxy > 0 and gold < 0:
    notes.append("Stark dollar och svagare guld: samma riktning som det klassiska sambandet.")
elif dxy > 0 and gold > 0:
    notes.append("Dollar och guld stiger samtidigt: något annat driver guld (oro, centralbanksköp).")
elif dxy < 0 and gold > 0:
    notes.append("Svagare dollar och starkare guld: klassiskt samband.")
if d10 > 0 and jpy > 0:
    notes.append("Stigande amerikansk ränta och starkare dollar mot yen: räntegapet drar USD/JPY uppåt.")
if jpy > 0 and jpy > 0 and df["USD/JPY"].iloc[-1] > 155:
    notes.append("USD/JPY över 155: historiskt en zon där japanska myndigheter har intervenerat "
                 "(risk för snabba rörelser åt andra hållet).")
if d10 > 0 and gold > 0:
    notes.append("Räntan stiger men guld håller sig: realräntesambandet fungerar inte just nu.")
for n_ in notes:
    st.write("• " + n_)

st.caption("Verktyget visar data och statistik. Det är inte investeringsrådgivning "
           "och garanterar inget om framtida rörelser.")
