text
# Singapore Hospital Bed Demand Forecast

This project is a small, beginner‑friendly healthcare analytics project using **Singapore public data** to analyse and forecast **hospital bed capacity**. 
It demonstrates an end‑to‑end workflow: data cleaning, reshaping, basic time‑series modelling (ARIMA), and visualisation in Python.

---

## 1. Project goals

- Load and clean official **“Beds in Inpatient Facilities”** data for Singapore.
- Reshape the dataset from **wide** (years as columns) to **long** (year, beds) format.
- Fit a simple **ARIMA time‑series model** to forecast bed demand for the next 5 years.
- Visualise historical vs forecasted bed capacity in a clear line chart.
- Document the steps similar to how a health‑analytics role would approach capacity planning.

---

## 2. Data

Source: Singapore public statistics (MOH / SingStat / data.gov.sg).

Example dataset used:
- **Beds In Inpatient Facilities, Annual** – CSV downloaded from the official site.
- File name in this repo (after download):  
  `BedsInInpatientFacilitiesAnnual.csv`

Key fields after cleaning:
- `DataSeries` – type of facility (e.g. “Acute Hospitals”, “Psychiatric Hospitals”).
- Year columns: `2006`, `2007`, …, `2023` → number of beds.

For modelling, we select one series (e.g. **“Acute Hospitals”**) and create a time series:
- `year` – integer year.
- `beds` – number of inpatient beds.

---

## 3. Methods

### 3.1. Environment

- Python 3.9+  
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `statsmodels`

### 3.2. Data preparation (wide → long)

Core steps:

1. **Read CSV**

```python
import pandas as pd

df = pd.read_csv("BedsInInpatientFacilitiesAnnual.csv")
Inspect available series

python
print(df.columns)
print(df["DataSeries"].unique())
Filter to one series (e.g. Acute Hospitals)

python
beds_row = df[df["DataSeries"] == "Acute Hospitals"].copy()
Convert from wide to long

python
beds_long = beds_row.melt(
    id_vars=["DataSeries"],
    var_name="year",
    value_name="beds"
)

beds_long["year"] = pd.to_numeric(beds_long["year"], errors="coerce")
beds_long["beds"] = pd.to_numeric(beds_long["beds"], errors="coerce")

beds_long = (
    beds_long
    .dropna(subset=["year", "beds"])
    .sort_values("year")
    .set_index("year")
)
Now beds_long is a proper time series with:

Index: year

Column: beds

4. **Time‑series forecasting**
4.1. Fit ARIMA
python
from statsmodels.tsa.arima.model import ARIMA

y = beds_long["beds"].astype("float64")

model = ARIMA(y, order=(1, 1, 1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=5)
y is the historical bed series.

order=(1,1,1) is a simple ARIMA configuration, suitable for a beginner‑level project.

forecast contains the next 5 years of predicted bed values.

4.2. Build combined series and plot python
import matplotlib.pyplot as plt

last_year = y.index.max()
future_years = range(last_year + 1, last_year + 1 + len(forecast))
forecast.index = future_years

plt.figure(figsize=(8, 4))

# Historical
plt.plot(y.index, y.values, label="Historical beds", marker="o")

# Forecast
plt.plot(forecast.index, forecast.values,
         label="Forecast beds", marker="o",
         linestyle="--", color="orange")

plt.xlabel("Year")
plt.ylabel("Beds in inpatient facilities")
plt.title("Singapore Acute Hospital Beds: History and 5‑Year Forecast")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
5. Repository structure
Example layout:

text
sg-hospital-bed-forecast/
├── data/
│   └── BedsInInpatientFacilitiesAnnual.csv   # raw data (not committed if restricted)
├── notebooks/
│   └── 01_preprocess_and_forecast.ipynb      # main analysis notebook
├── src/
│   └── preprocess.py                         # optional: reusable ETL functions
├── plots/
│   └── beds_forecast.png                     # saved chart
└── README.md                                 # this file
6. How to run
Clone the repo

bash
git clone https://github.com/<your-username>/sg-hospital-bed-forecast.git
cd sg-hospital-bed-forecast
Create environment & install dependencies

bash
pip install -r requirements.txt
(Or manually install: pandas, numpy, matplotlib, statsmodels.)

Add the dataset

Download the CSV from the official source.

Save as BedsInInpatientFacilitiesAnnual.csv under data/.

Run the notebook

Open notebooks/01_preprocess_and_forecast.ipynb in Jupyter / VS Code.

Run all cells to:

Clean and reshape data.

Fit the ARIMA model.

Generate the forecast plot.

7. Interpretation & next steps
This is intentionally a beginner–intermediate project:

It shows how a healthcare data analyst could start exploring capacity planning for hospital beds.

It uses a simple ARIMA model; in a real setting, you’d pair bed data with:

admissions,

length of stay,

population ageing trends.

Possible extensions:

Compare different facility types (acute vs community vs nursing homes).

Use more advanced models (e.g. SARIMAX with exogenous variables).

Wrap the pipeline into a scheduled job and publish results to a dashboard tool.

## 9. **Prophet model and ARIMA vs Prophet comparison**

In addition to ARIMA, this project also uses **Prophet** (additive time-series model) to forecast hospital bed demand and compare results.

### 9.1. Prophet setup

Prophet expects a DataFrame with:

- `ds` – date/time (here we convert the year to a date, e.g. `YYYY-01-01`)
- `y` – value to forecast (beds)

Example preparation:

```python
from prophet import Prophet
import pandas as pd

# beds_long has index = year, column = beds
prophet_df = beds_long.reset_index().rename(columns={"year": "ds", "beds": "y"})

# Convert ds into a proper datetime (year -> 1 Jan of that year)
prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], format="%Y")
Fit Prophet and forecast the next 5 years:

python
m = Prophet()
m.fit(prophet_df)

future = m.make_future_dataframe(periods=5, freq="Y")
forecast_prophet = m.predict(future)

# Keep only the forecast horizon
prophet_future = forecast_prophet.tail(5)[["ds", "yhat"]]
9.2. ARIMA vs Prophet: combined plot
We align the ARIMA and Prophet forecasts for visual comparison:

python
import matplotlib.pyplot as plt

# ARIMA series y (historical) and forecast (5 years) already computed
# y index: year (int), forecast index: future years (int)

# Convert ARIMA index to datetime for plotting alongside Prophet
y_arima = y.copy()
y_arima.index = pd.to_datetime(y_arima.index, format="%Y")

forecast_arima = forecast.copy()
forecast_arima.index = pd.to_datetime(forecast_arima.index, format="%Y")

plt.figure(figsize=(9, 5))

# Historical beds
plt.plot(y_arima.index, y_arima.values, label="Historical beds", marker="o", color="C0")

# ARIMA forecast
plt.plot(forecast_arima.index, forecast_arima.values,
         label="ARIMA forecast", marker="o", linestyle="--", color="C1")

# Prophet forecast
plt.plot(prophet_future["ds"], prophet_future["yhat"],
         label="Prophet forecast", marker="s", linestyle="--", color="C2")

plt.xlabel("Year")
plt.ylabel("Beds in inpatient facilities")
plt.title("Singapore Hospital Beds: ARIMA vs Prophet (5‑Year Forecast)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
9.3. Interpretation
This comparison plot makes it easy to see how:

ARIMA extrapolates based mainly on recent linear trends and differencing.

Prophet fits a smooth trend (and could handle seasonality if present), which can sometimes give slightly different trajectories.

For this beginner–intermediate project, the goal is not to prove which model is “best”, but to:

Demonstrate how to fit and visualise two forecasting approaches on the same healthcare capacity metric.

Start a discussion on model choice and uncertainty in hospital planning and resource allocation.

8. Disclaimer
Data used here are aggregate public statistics, not patient‑level data.

This project is for educational/portfolio purposes only and should not be used for real‑world clinical or operational decisions without appropriate validation and domain review.

Done by Rupali Rajesh Desai as part of her portfolio build up. 
