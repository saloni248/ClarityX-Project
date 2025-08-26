# 🏗️ Test-Project: Asset Valuation & Analysis

## 📌 Project Overview
This project focuses on **asset valuation, exploratory analytics, and predictive modeling** using two datasets.  
It integrates **statistical methods, GIS spatial analysis, and machine learning models** to estimate asset values, build an interactive dashboard, and classify assets into meaningful categories.

---

## 🎯 Objectives & Tasks

### ✅ Task 1: Asset Valuation
- Estimate values of assets in **Dataset 1** using information from **Dataset 2**.
- Develop a **valuation model** that captures:
  - Market-driven price variations.
  - Asset-level features (size, type, year, etc.).
  - Spatial/geographical influences (region, ZIP codes).
- Evaluate the model using metrics such as **R², RMSE, and MAE**.

---

### ✅ Task 2: Analytical Dashboard
- Build an **interactive dashboard** for asset insights.
- Include **Descriptive Statistics**:
  - Price distributions, trends, and utilization levels.
  - Regional/cluster-level summaries.
- Apply **Inferential Statistics**:
  - Hypothesis testing (metro vs non-metro, old vs new assets, etc.).
  - Confidence intervals for valuation estimates.
- Utilize **GIS libraries** (e.g., Folium, GeoPandas) to:
  - Visualize spatial distribution of assets.
  - Identify metro vs periphery divides.
  - Map hotspots of growth or decline.

---

### ✅ Task 3: Machine Learning Models
- **Unsupervised Learning:**
  - Apply clustering (e.g., K-Means, DBSCAN) to create asset classes based on price, location, and features.
  - Identify high-performing vs underperforming clusters.
- **Supervised Learning:**
  - Build predictive models (e.g., Random Forest, Gradient Boosting) to classify and predict assets by valuation category.
  - Use the valuation model from Task 1 as the target variable for classification.

---

## 🛠️ Tech Stack & Libraries
- **Languages:** Python 3.x  
- **Data Analysis:** Pandas, NumPy, Scikit-learn, Statsmodels  
- **Visualization:** Matplotlib, Seaborn, Plotly, Folium  
- **GIS Analysis:** GeoPandas, Shapely, Folium  
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM  
- **Dashboarding:** Streamlit  

---

## 📊 Expected Deliverables
1. **Valuation Model** for Dataset 1 assets.  
2. **Interactive Analytical Dashboard** with descriptive + spatial analytics.  
3. **Clustering & Classification Models** for asset segmentation and prediction.  
4. **Comprehensive Report** documenting methodology, findings, and recommendations.

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone <repo-link>
   cd Test-Project

2. Install dependencies:

pip install -r requirements.txt


3. Run the Jupyter notebooks for Task 1, Task 2, and Task 3:

jupyter notebook


4. Launch the dashboard:

streamlit run dashboard.py
