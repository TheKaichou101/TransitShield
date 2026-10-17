
import pandas as pd
import streamlit as st

from datetime import date, time, datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

st.set_page_config(page_title="TransitShield", layout="wide")

st.title("TransitShield")
st.write("AI-powered bus delay prediction")

# Load the dataset
data = pd.read_csv("transit.csv")

# Convert the date and departure columns
data["date"] = pd.to_datetime(data["date"], errors="coerce")
data["expected_departure"] = pd.to_datetime(
    data["expected_departure"], errors="coerce"
)
data["actual_departure"] = pd.to_datetime(
    data["actual_departure"], errors="coerce"
)

# Remove rows without essential information
data = data.dropna(
    subset=["stop_id", "stop_name", "expected_departure"]
).copy()

# Calculate delay in minutes
data["delay_minutes"] = (
    data["actual_departure"] - data["expected_departure"]
).dt.total_seconds() / 60

# Treat missing actual departure times as cancellations.
# This assumes missing values really mean cancelled services.
data["disrupted"] = (
    data["actual_departure"].isna()
    | (data["delay_minutes"] >= 5)
).astype(int)

# Create features for the model
data["hour"] = data["expected_departure"].dt.hour
data["weekday"] = data["expected_departure"].dt.dayofweek

data["stop_id"] = data["stop_id"].astype(str)

features = ["stop_id", "hour", "weekday"]
target = "disrupted"

st.subheader("Historical transport data")

st.metric("Total records", len(data))
st.metric("Disrupted records", int(data[target].sum()))

st.dataframe(
    data[[
        "stop_id",
        "stop_name",
        "date",
        "expected_departure",
        "actual_departure",
        "passenger_count",
        "delay_minutes",
        "disrupted"
    ]].head(20)
)

# Check that the data contains both outcomes
if data[target].nunique() < 2:
    st.error(
        "The dataset needs both disrupted and non-disrupted records "
        "to train this model."
    )
    st.stop()

# Prepare training and testing data
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert stop IDs into values the model can understand
preparation = ColumnTransformer([
    (
        "stop",
        OneHotEncoder(handle_unknown="ignore"),
        ["stop_id"]
    )
], remainder="passthrough")

# Create and train the model
model = Pipeline([
    ("prepare", preparation),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
recall = recall_score(
    y_test, predictions, zero_division=0
)

st.subheader("Model evaluation")

col1, col2 = st.columns(2)
col1.metric("Test accuracy", f"{accuracy:.1%}")
col2.metric("Disruption recall", f"{recall:.1%}")

st.caption(
    "These results come from the selected test split. "
    "They do not guarantee future prediction accuracy."
)

# Display stops with the most historical disruptions
st.subheader("Stops with the most disruptions")

stop_summary = (
    data.groupby(["stop_id", "stop_name"])
    .agg(
        total_services=("disrupted", "count"),
        disruptions=("disrupted", "sum")
    )
    .reset_index()
)

stop_summary["disruption_rate"] = (
    stop_summary["disruptions"]
    / stop_summary["total_services"] * 100
)

stop_summary = stop_summary.sort_values(
    "disruption_rate", ascending=False
)

st.dataframe(stop_summary)

# Predict the risk for a future service
st.subheader("Predict disruption risk")

st.write(
    "Choose a stop and departure time to estimate disruption risk."
)

stops = (
    data[["stop_id", "stop_name"]]
    .drop_duplicates()
    .sort_values("stop_name")
)

stop_options = {
    f"{row.stop_name} (ID: {row.stop_id})": row.stop_id
    for row in stops.itertuples(index=False)
}

selected_stop = st.selectbox(
    "Select stop", list(stop_options.keys())
)

selected_date = st.date_input(
    "Expected departure date", date.today()
)

selected_time = st.time_input(
    "Expected departure time", time(8, 30)
)

if st.button("Predict risk"):

    departure = datetime.combine(
        selected_date, selected_time
    )

    new_service = pd.DataFrame([{
        "stop_id": stop_options[selected_stop],
        "hour": departure.hour,
        "weekday": departure.weekday()
    }])

    probabilities = model.predict_proba(new_service)[0]

    disruption_index = list(model.classes_).index(1)
    risk = probabilities[disruption_index]

    st.metric("Estimated disruption risk", f"{risk:.1%}")

    if risk >= 0.7:
        st.warning(
            "High estimated risk. Consider reviewing this service."
        )
    elif risk >= 0.4:
        st.info(
            "Moderate estimated risk. Further review may be useful."
        )
    else:
        st.success(
            "Lower estimated risk. Continue normal monitoring."
        )
