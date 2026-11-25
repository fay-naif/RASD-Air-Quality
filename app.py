import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from core import PREDICTORS, load_real_data, prepare_data


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="راصد | RASD" if st.session_state.get("language", "العربية") == "العربية" else "RASD | راصد",
    page_icon=":material/radar:",
    layout="wide",
)

st.session_state.setdefault("language", "العربية")
st.session_state.setdefault("appearance", "داكن")
st.session_state.setdefault("classification_result", None)
st.session_state.setdefault("direct_readings", {})

with st.sidebar:
    st.markdown(
        "<div class='brand-lockup'><span class='brand-mark'>R</span>"
        "<div><strong>راصد</strong><small>RASD</small></div></div>",
        unsafe_allow_html=True,
    )
    st.segmented_control(
        "اللغة / Language",
        ["العربية", "English"],
        key="language",
        persist_state="session",
    )
    st.segmented_control(
        "المظهر / Appearance",
        ["فاتح", "داكن"],
        key="appearance",
        persist_state="session",
    )
    st.caption("بيانات تاريخية · محطة بكين" if st.session_state["language"] == "العربية" else "Historical data · Beijing station")

ar = st.session_state["language"] == "العربية"
dark = st.session_state["appearance"] == "داكن"


def tr(arabic, english):
    return arabic if ar else english


AR_TO_EN = {
    "منخفض": "Low",
    "مرتفع قليلاً": "Elevated",
    "مرتفع": "High",
    "مرتفع جداً": "Very high",
}


bg = "#091411" if dark else "#F3F7F5"
surface = "#12231F" if dark else "#FFFFFF"
surface2 = "#183029" if dark else "#EAF2EE"
text = "#EAF4F1" if dark else "#17312D"
muted = "#9FB2AB" if dark else "#647A73"
accent = "#64D8C4" if dark else "#087F72"
line = "#29423B" if dark else "#D7E4DF"
warm = "#E7B86D" if dark else "#D69A46"

st.html(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Tajawal:wght@400;500;600;700&display=swap');
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
      direction: {"rtl" if ar else "ltr"};
      font-family: 'Tajawal','Inter',sans-serif !important;
    }}
    [data-testid="stSidebar"] {{ direction: {"rtl" if ar else "ltr"}; }}
    [data-testid="stSidebar"], [data-testid="stSidebar"] * {{
      color: {text} !important;
    }}
    [data-testid="stSidebar"] {{
      background: {surface2} !important;
      border-{"left" if ar else "right"}: 1px solid {line} !important;
    }}
    [data-testid="stSidebarContent"] {{
      padding-top: 1.1rem;
    }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
      gap: 0.75rem;
    }}
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] caption {{
      color: {text};
    }}
    [data-testid="stAppViewContainer"] * {{ scrollbar-color: {accent} transparent; }}
    [data-testid="stAppViewContainer"] {{
      background:
        radial-gradient(circle at 88% 8%, {"rgba(100,216,196,.10)" if dark else "rgba(8,127,114,.09)"}, transparent 24rem),
        {bg};
    }}
    .block-container {{ max-width: 1180px; padding-top: 1.25rem; padding-bottom: 3rem; }}
    [data-testid="stMain"] {{
      background: transparent;
    }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stMetric"] {{
      background: {surface};
      border: 1px solid {line};
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 10px 32px {"rgba(0,0,0,.16)" if dark else "rgba(23,49,45,.06)"};
      animation: rasd-in .55s cubic-bezier(.2,.8,.2,1) both;
    }}
    [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {{ color: {muted} !important; }}
    [data-testid="stMetricValue"] {{ color: {text} !important; font-weight: 700; }}
    [data-testid="stMetric"] {{
      backdrop-filter: blur(12px);
    }}
    [data-testid="stForm"] {{
      border-radius: 22px;
    }}
    [data-testid="stDataFrame"], [data-testid="stTable"], [data-testid="stVegaLiteChart"], [data-testid="stPlotlyChart"] {{
      background: {surface};
      border: 1px solid {line};
      border-radius: 16px;
      overflow: hidden;
    }}
    [data-testid="stForm"] input,
    [data-testid="stForm"] [role="spinbutton"],
    [data-testid="stForm"] [data-baseweb="input"] {{
      background: {surface} !important;
      color: {text} !important;
      border-color: {line} !important;
    }}
    [data-testid="stForm"] label {{
      color: {muted} !important;
      font-weight: 600;
    }}
    [data-testid="stTabs"] [data-baseweb="tab-list"] {{
      border-bottom: 1px solid {line};
      gap: 8px;
      margin-bottom: 20px;
      align-items: end;
    }}
    [data-testid="stTabs"] button[role="tab"] {{
      padding: 10px 14px;
      border-radius: 10px 10px 0 0;
      font-weight: 650;
    }}
    [data-testid="stTabs"] button[aria-selected="true"] {{
      background: {"rgba(100,216,196,.09)" if dark else "rgba(8,127,114,.08)"};
      color: {accent} !important;
    }}
    [data-testid="stForm"], [data-testid="stFormSubmitButton"] {{
      border-radius: 18px;
    }}
    [data-testid="stForm"] {{
      background: {surface};
      border: 1px solid {line};
      padding: 20px;
      box-shadow: 0 14px 38px {"rgba(0,0,0,.12)" if dark else "rgba(23,49,45,.05)"};
    }}
    .stButton button, [data-testid="stFormSubmitButton"] button {{
      min-height: 44px;
      border-radius: 12px;
      font-weight: 700;
      transition: transform .18s ease, box-shadow .18s ease, background .18s ease;
    }}
    .stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {{
      transform: translateY(-2px);
      box-shadow: 0 10px 24px {"rgba(100,216,196,.18)" if dark else "rgba(8,127,114,.16)"};
    }}
    .hero {{
      position: relative;
      overflow: hidden;
      background: linear-gradient(125deg, {surface2}, {surface});
      border: 1px solid {line};
      border-radius: 26px;
      padding: 30px 34px;
      margin: 4px 0 22px;
      animation: rasd-in .65s cubic-bezier(.2,.8,.2,1) both;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      width: 180px;
      height: 180px;
      {"left" if ar else "right"}: -75px;
      bottom: -100px;
      border-radius: 50%;
      border: 38px solid {"rgba(100,216,196,.10)" if dark else "rgba(8,127,114,.10)"};
      animation: rasd-pulse 7s ease-in-out infinite;
    }}
    .brand-lockup {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 2px 0 14px;
      direction: {"rtl" if ar else "ltr"};
    }}
    .brand-lockup strong {{ display: block; font-size: 20px; line-height: 1.05; }}
    .brand-lockup small {{ display: block; color: {accent}; font-size: 11px; letter-spacing: 2px; margin-top: 4px; }}
    .brand-mark {{
      width: 42px; height: 42px; display: grid; place-items: center;
      border-radius: 14px; background: {accent}; color: {surface};
      font-family: 'Inter',sans-serif; font-size: 20px; font-weight: 800;
      box-shadow: 0 8px 20px {"rgba(100,216,196,.20)" if dark else "rgba(8,127,114,.18)"};
    }}
    .hero-kicker {{ color: {accent}; font-size: 12px; font-weight: 700; letter-spacing: 1.4px; }}
    .hero-title {{ font-size: clamp(30px,4vw,48px); margin: 8px 0 5px; color: {text}; line-height: 1.2; }}
    .hero-copy {{ max-width: 680px; color: {muted}; font-size: 16px; line-height: 1.8; }}
    .status-pill {{
      display: inline-flex; align-items: center; gap: 7px;
      margin-top: 12px; padding: 7px 11px; border-radius: 999px;
      background: {"rgba(100,216,196,.10)" if dark else "rgba(8,127,114,.08)"};
      color: {accent}; border: 1px solid {line}; font-size: 12px; font-weight: 700;
    }}
    .section-card {{
      background: {surface}; border: 1px solid {line}; border-radius: 18px;
      padding: 18px 20px; margin: 8px 0 14px;
      animation: rasd-in .55s cubic-bezier(.2,.8,.2,1) both;
    }}
    [data-testid="stAlert"] {{
      border-radius: 14px;
      border-right-width: 4px;
      animation: rasd-in .45s ease both;
    }}
    @keyframes rasd-in {{
      from {{ opacity: 0; transform: translateY(10px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes rasd-pulse {{
      0%,100% {{ transform: scale(.96); opacity: .7; }}
      50% {{ transform: scale(1.06); opacity: 1; }}
    }}
    @media (max-width: 700px) {{
      .block-container {{ padding: 1rem .65rem 2rem; }}
      .hero {{ padding: 22px 20px; border-radius: 20px; }}
      .hero-title {{ font-size: 30px; }}
      [data-testid="stMetric"] {{ padding: 14px; min-height: 112px; }}
      [data-testid="stTabs"] button[role="tab"] {{ padding: 9px 9px; font-size: 13px; }}
      [data-testid="stForm"] {{ padding: 14px; }}
    }}
    @media (prefers-reduced-motion: reduce) {{
      *, *::before, *::after {{ animation: none !important; transition: none !important; }}
    }}
    </style>
    """
)


@st.cache_data(show_spinner=False)
def load_app_data():
    return prepare_data(load_real_data())


@st.cache_resource(show_spinner=False)
def load_app_models():
    return (
        joblib.load(ROOT / "results" / "model.joblib"),
        joblib.load(ROOT / "results" / "classification_model.joblib"),
    )


@st.cache_data(show_spinner=False)
def training_medians():
    return classification_training_medians


try:
    data = load_app_data()
    forecast_bundle, classification_bundle = load_app_models()
    classification_training_medians = classification_bundle["training_medians"]
except Exception as exc:
    st.error(tr(f"تعذر تحميل المشروع: {exc}", f"Project could not be loaded: {exc}"))
    st.stop()

latest = data.iloc[-1]
previous = data.iloc[-2]
try:
    forecast = float(
        forecast_bundle["model"].predict(
            pd.DataFrame([latest], columns=forecast_bundle["features"])
        )[0]
    )
except Exception as exc:
    forecast = None
    forecast_error = exc

st.markdown(
    f"""
    <div class="hero">
      <div class="hero-kicker">{tr("مستشعرات بيئية · رؤية أوضح للساعة القادمة","ENVIRONMENTAL INTELLIGENCE · NEXT-HOUR OUTLOOK")}</div>
      <h1 class="hero-title">{tr("راصد يتابع جودة الهواء","RASD monitors air quality")}</h1>
      <div class="hero-copy">{tr("تحليل تاريخي وتوقعات وتصنيف مفسر لتركيز PM2.5، بتجربة هادئة وواضحة.","Historical analysis, forecasting and explainable PM2.5 classification in one clear workspace.")}</div>
      <div class="status-pill"><span>●</span>{tr("بيانات المحطة التاريخية متاحة","Historical station data available")}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

overview, estimate, evaluation, method = st.tabs(
    [
        tr("النظرة العامة", "Overview"),
        tr("التصنيف التفاعلي", "Interactive classification"),
        tr("نتائج التقييم", "Evaluation"),
        tr("المنهجية", "Method"),
    ]
)

with overview:
    pm25 = float(latest["PM2.5"])
    pm10 = float(latest["PM10"])
    delta = pm25 - float(previous["PM2.5"])

    st.subheader(
        tr("آخر قراءات المحطة", "Latest station readings"),
        icon=":material/analytics:",
    )
    with st.container(horizontal=True):
        st.metric(
            "PM2.5",
            f"{pm25:,.1f}",
            f"{delta:+.1f}",
            delta_description="µg/m³",
            border=True,
            height="stretch",
        )
        st.metric("PM10", f"{pm10:,.1f}", "µg/m³", border=True, height="stretch")
        st.metric(
            tr("توقع الساعة التالية", "Next-hour forecast"),
            f"{forecast:,.1f}" if forecast is not None else "—",
            tr("ميكروغرام/م³", "µg/m³"),
            border=True,
            height="stretch",
        )
        if forecast is None:
            st.error(tr(
                f"تعذر تشغيل التنبؤ: {forecast_error}",
                f"Forecast could not be generated: {forecast_error}",
            ))

    with st.container(border=True):
        st.subheader(
            tr("اتجاه الملوثات", "Pollution trend"),
            icon=":material/show_chart:",
        )
        chart = data.tail(168)[["timestamp", "PM2.5", "PM10"]].set_index("timestamp")
        st.line_chart(
            chart,
            color=[accent, warm],
            height=320,
            alt=tr("اتجاه PM2.5 وPM10 خلال آخر 168 ساعة", "PM2.5 and PM10 trend over 168 hours"),
        )

    st.caption(
        tr(
            "بيانات تاريخية لأغراض البحث والاستكشاف؛ ليست تنبيهًا مباشرًا أو نصيحة صحية.",
            "Historical research data; not a live alert or health guidance.",
        )
    )

with estimate:
    st.subheader(
        tr("اختبر تصنيف الساعة التالية", "Estimate the next-hour class"),
        icon=":material/tune:",
    )
    st.caption(
        tr(
            "النتيجة تستخدم القراءات المباشرة ومدخل تاريخ ووقت واضحين. "
            "لا توجد تأخرات أو متوسطات متحركة في هذا النموذج؛ الخانات المباشرة الفارغة تعوّض بوسيط مجموعة التدريب فقط.",
            "The result uses direct readings and an explicit date and time. "
            "This model does not use lags or rolling means; blank direct-reading fields use training-set medians only.",
        )
    )

    fields = [
        ("PM2.5", "PM2.5 (µg/m³)", 0.0, 1000.0),
        ("PM10", "PM10 (µg/m³)", 0.0, 1500.0),
        ("SO2", "SO2 (µg/m³)", 0.0, 1000.0),
        ("NO2", "NO2 (µg/m³)", 0.0, 1000.0),
        ("CO", "CO (µg/m³)", 0.0, 10000.0),
        ("O3", "O3 (µg/m³)", 0.0, 1000.0),
        ("TEMP", tr("الحرارة (°م)", "Temperature (°C)"), -50.0, 60.0),
        ("PRES", tr("الضغط (hPa)", "Pressure (hPa)"), 800.0, 1100.0),
        ("DEWP", tr("درجة الندى (°م)", "Dew point (°C)"), -50.0, 60.0),
        ("RAIN", tr("الأمطار (مم)", "Rain (mm)"), 0.0, 500.0),
        ("WSPM", tr("سرعة الرياح (م/ث)", "Wind speed (m/s)"), 0.0, 100.0),
    ]
    values = {}

    def update_direct_reading(key):
        value = st.session_state.get(f"reading_{key}")
        st.session_state["direct_readings"][key] = value

    with st.container(border=True):
        cols = st.columns(3)
        for index, (key, label, minimum, maximum) in enumerate(fields):
            with cols[index % 3]:
                values[key] = st.number_input(
                    label,
                    min_value=minimum,
                    max_value=maximum,
                    value=None,
                    key=f"reading_{key}",
                    persist_state="session",
                    on_change=update_direct_reading,
                    args=(key,),
                )
        observation_date = st.date_input(
            tr("تاريخ القراءة", "Observation date"),
            value=latest["timestamp"].date(),
            key="observation_date",
            persist_state="session",
        )
        hour = st.number_input(
            tr("ساعة اليوم", "Hour of day"),
            0,
            23,
            value=int(latest["timestamp"].hour),
            key="observation_hour",
            persist_state="session",
        )
        submitted = st.button(
            tr("تحليل التصنيف", "Analyze classification"),
            icon=":material/auto_awesome:",
            width="stretch",
            type="primary",
        )

    if submitted:
        try:
            medians = training_medians()
            direct_values = {
                key: st.session_state["direct_readings"].get(key)
                for key, _label, _minimum, _maximum in fields
            }
            has_new_reading = any(value is not None for value in direct_values.values())
            if has_new_reading:
                row = pd.Series(medians, dtype="float64")
                for key, value in direct_values.items():
                    if value is not None:
                        row[key] = float(value)
                source_type = "user"
                source_label = tr(
                    "قراءات مستخدم متاحة مع تعويض حقول القراءات المباشرة بوسيط التدريب فقط",
                    "Available user readings; only direct-reading blanks use training medians",
                )
                observation_timestamp = datetime.combine(
                    st.session_state.get("observation_date", latest["timestamp"].date()),
                    datetime.min.time(),
                ).replace(hour=int(st.session_state.get("observation_hour", latest["timestamp"].hour)))
            else:
                row = pd.Series(medians, dtype="float64")
                for key in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]:
                    if key in latest:
                        row[key] = float(latest[key])
                source_type = "historical"
                source_label = tr(
                    f"قراءة تاريخية أصلية من المحطة: {latest['timestamp']}",
                    f"Original historical station reading: {latest['timestamp']}",
                )
                observation_timestamp = latest["timestamp"]
            row["hour_sin"] = np.sin(2 * np.pi * observation_timestamp.hour / 24)
            row["hour_cos"] = np.cos(2 * np.pi * observation_timestamp.hour / 24)
            row["month_sin"] = np.sin(2 * np.pi * observation_timestamp.month / 12)
            row["month_cos"] = np.cos(2 * np.pi * observation_timestamp.month / 12)
            frame = pd.DataFrame([row], columns=classification_bundle["features"])
            model = classification_bundle["model"]
            predicted_ar = str(model.predict(frame)[0])
            labels = list(model.classes_)
            probabilities = model.predict_proba(frame)[0]
            confidence = float(probabilities[labels.index(predicted_ar)])
            displayed_label = predicted_ar if ar else AR_TO_EN[predicted_ar]
            st.session_state["classification_result"] = {
                "predicted_ar": predicted_ar,
                "labels": labels,
                "probabilities": probabilities.tolist(),
                "confidence": confidence,
                "source_type": source_type,
                "source_label_ar": (
                    f"قراءة تاريخية أصلية من المحطة: {latest['timestamp']}"
                    if source_type == "historical"
                    else "قراءات مستخدم متاحة؛ حقول القراءات المباشرة الفارغة تعوّض بوسيط التدريب"
                ),
                "source_label_en": (
                    f"Original historical station reading: {latest['timestamp']}"
                    if source_type == "historical"
                    else "Available user readings; blank direct-reading fields use training medians"
                ),
            }
        except Exception as exc:
            st.error(
                tr(
                    f"تعذر إتمام التصنيف بهذه المدخلات: {exc}",
                    f"Classification could not be completed with these inputs: {exc}",
                )
            )

    if st.session_state["classification_result"] is not None:
        result = st.session_state["classification_result"]
        predicted_ar = result["predicted_ar"]
        labels = result["labels"]
        probabilities = np.asarray(result["probabilities"], dtype=float)
        confidence = float(result["confidence"])
        displayed_label = predicted_ar if ar else AR_TO_EN[predicted_ar]
        source_label = (
            result["source_label_ar"] if ar else result["source_label_en"]
        )
        st.success(
            tr(
                f"الفئة المتوقعة: {displayed_label}",
                f"Predicted class: {displayed_label}",
            ),
            icon=":material/check_circle:",
        )
        st.caption(
            tr(
                f"ثقة النموذج للنتيجة المختارة: {confidence:.1%}",
                f"Model confidence for the selected result: {confidence:.1%}",
            )
        )
        st.info(
            tr(f"مصدر المدخلات: {source_label}", f"Input source: {source_label}")
        )
        probability_table = pd.DataFrame(
            {
                tr("الفئة", "Class"): [
                    label if ar else AR_TO_EN[label] for label in labels
                ],
                tr("الاحتمال", "Probability"): probabilities,
            }
        )
        st.bar_chart(
            probability_table,
            x=tr("الفئة", "Class"),
            y=tr("الاحتمال", "Probability"),
            color=accent,
            height=260,
            alt=tr("احتمالات فئات التصنيف", "Classification probability chart"),
        )

with evaluation:
    metrics = classification_bundle["test_metrics"]
    st.subheader(
        tr("اختبار معزول زمنيًا", "Chronologically isolated test"),
        icon=":material/verified:",
    )
    c1, c2, c3 = st.columns(3)
    c1.metric(tr("الدقة", "Accuracy"), f"{metrics['accuracy']:.1%}", border=True)
    c2.metric(
        tr("الدقة المتوازنة", "Balanced accuracy"),
        f"{metrics['balanced_accuracy']:.1%}",
        border=True,
    )
    c3.metric("Macro F1", f"{metrics['macro_f1']:.1%}", border=True)

    st.caption(
        tr(
            f"{classification_bundle['splits']['test']:,} ساعة في الاختبار النهائي.",
            f"{classification_bundle['splits']['test']:,} hours in the final test.",
        )
    )

    with st.container(border=True):
        st.subheader(tr("مصفوفة الالتباس", "Confusion matrix"))
        display_classes = (
            classification_bundle["classes"]
            if ar
            else [AR_TO_EN[label] for label in classification_bundle["classes"]]
        )
        st.dataframe(
            pd.DataFrame(
                classification_bundle["confusion_matrix"],
                index=display_classes,
                columns=display_classes,
            ),
            hide_index=False,
            width="stretch",
        )

    report = json.loads(
        (ROOT / "results" / "metrics.json").read_text(encoding="utf-8")
    )
    tm = report["held_out_test"]
    st.subheader(
        tr("تقييم التنبؤ العددي", "Numerical forecast evaluation"),
        icon=":material/query_stats:",
    )
    f1, f2, f3 = st.columns(3)
    f1.metric("MAE", f"{tm['MAE']:.3f}", border=True)
    f2.metric("RMSE", f"{tm['RMSE']:.3f}", border=True)
    f3.metric("R²", f"{tm['R2']:.4f}", border=True)

with method:
    st.subheader(
        tr("كيف يعمل راصد؟", "How RASD works"),
        icon=":material/hub:",
    )
    with st.container(border=True):
        st.markdown(
            tr(
                "**المتغير المستهدف**  \nPM2.5 بعد ساعة واحدة، مفروز إلى أربع فئات: منخفض، مرتفع قليلًا، مرتفع، مرتفع جدًا.",
                "**Target variable**  \nNext-hour PM2.5, grouped into low, elevated, high and very high classes.",
            )
        )
    with st.container(border=True):
        st.markdown(
            tr(
                "**منع تسرب البيانات**  \nالمدخلات مقتصرة على القراءات الحالية والسابقة والإشارات الزمنية؛ لا يدخل الهدف المستقبل في التدريب.",
                "**Leakage control**  \nOnly current/past readings and time signals enter the model; next-hour targets are excluded.",
            )
        )
    split = classification_bundle["splits"]
    with st.container(border=True):
        st.markdown(
            tr(
                f"**تقسيم زمني**  \nتدريب {split['train']:,} ساعة · تحقق {split['validation']:,} ساعة · اختبار {split['test']:,} ساعة.",
                f"**Chronological split**  \nTraining {split['train']:,} hours · validation {split['validation']:,} hours · test {split['test']:,} hours.",
            )
        )
