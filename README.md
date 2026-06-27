# 🌍 Geospatial Business Intelligence Dashboard  
### Retail Expansion Opportunity & Market Coverage Analysis

---

## 📌 Project Overview

This project performs **Geospatial Data Analysis** to identify optimal cities for business expansion using spatial modeling and demand-gap detection.

The system analyzes:

- 📍 Existing store locations  
- 📊 Customer demand density  
- 📈 Market demand index  
- 📏 Distance gaps from nearest stores  
- 💰 Store performance clusters  
- 🎯 Expansion opportunity scoring  

It generates:

✅ A multi-panel geospatial dashboard  
✅ Ranked expansion targets  
✅ Coverage gap analysis  
✅ Strategic recommendation report  

---

## 🖥️ Dashboard Output

Running the script generates:

```
geospatial_business_analysis.png
```

The dashboard contains 7 analytical panels:

---

### 1️⃣ Geographic Business Presence & Expansion Opportunities
- Existing store locations
- Opportunity tier markers
- Sales-based color scaling
- Geographic distribution mapping

---

### 2️⃣ Top Expansion Targets
- Ranked by Opportunity Score
- Tier classification (Strong / Moderate / Low)
- Visual comparison against threshold

---

### 3️⃣ Customer Demand vs Store Coverage Gaps
- Demand intensity heatmap
- Underserved high-demand zones
- Coverage overlap visualization

---

### 4️⃣ Store Performance Clusters
- K-Means clustering
- Zone-based performance analysis
- Average sales & ratings per cluster

---

### 5️⃣ Demand vs Presence Gap Matrix
Quadrant-based decision support:

- ✅ Well Served  
- 🎯 Prime Opportunity  
- ⚠ Saturate First  
- ❌ Low Priority  

---

### 6️⃣ Underserved Market Gap Analysis
- Demand Index vs Proximity Score comparison
- Normalized scoring visualization

---

### 7️⃣ Analytics Summary KPIs
- Active Stores
- Monthly Revenue
- Prime Opportunities
- Average Gap Distance
- #1 Target Market
- Market Coverage %

---

## 📊 Executive Summary (Current Run)

- **Total Markets Analyzed:** 40  
- **Existing Store Locations:** 76  
- **Prime Expansion Targets:** 0  
- **Strong Expansion Targets:** 15  
- **Current Monthly Revenue:** $45.9M  
- **Average Store Monthly Sales:** $604K  

---

## 🎯 Top Expansion Targets

| Rank | City        | Score | Tier   | Gap (km) |
|------|------------|-------|--------|----------|
| 1    | Seattle    | 0.600 | STRONG | 1094 |
| 2    | Portland   | 0.547 | STRONG | 863  |
| 3    | New York   | 0.542 | STRONG | 5    |
| 4    | Phoenix    | 0.541 | STRONG | 560  |
| 5    | Denver     | 0.537 | STRONG | 1165 |

---

## 🏪 Store Cluster Performance

| Cluster | Stores | Avg Sales | Total Sales | Avg Rating |
|----------|--------|------------|-------------|------------|
| Zone‑1 | 45 | $711K | $32.0M | ⭐⭐⭐⭐ 4.0 |
| Zone‑2 | 16 | $556K | $8.9M | ⭐⭐⭐ 3.8 |
| Zone‑0 | 15 | $335K | $5.0M | ⭐⭐⭐ 3.8 |

---

## 🧠 Methodology

### 1️⃣ Spatial Analysis
- KD-Tree for nearest-store lookup
- Distance-based coverage gap calculation
- Geographic coordinate modeling

### 2️⃣ Demand Modeling
- Synthetic demand generation
- Gaussian smoothing for heatmap
- Demand intensity scoring

### 3️⃣ Opportunity Score Formula

```
Opportunity Score =
  40% Market Demand
+ 35% Distance Gap
+ 25% Median Income
```

Higher score = Higher expansion priority.

---

### 4️⃣ Clustering
- K-Means clustering
- Silhouette score optimization
- Performance-based grouping

---

## 🛠️ Technologies Used

- Python 3.13
- NumPy
- Pandas
- Matplotlib
- SciPy
- Scikit-learn

---

## 📦 Requirements

Create a file named:

```
requirements.txt
```

Add:

```
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.8.0
scipy>=1.11.0
scikit-learn>=1.4.0
```

Install using:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1: Navigate to Project Folder

```powershell
cd "C:\Users\aryan\OneDrive\Desktop\Geospatial Data Analysis for Business Expansion"
```

### Step 2: Run the Script

```powershell
python data.py
```

OR

```powershell
& "C:\Program Files\Python313\python.exe" "data.py"
```

---

## 📁 Project Structure

```
Geospatial Data Analysis for Business Expansion/
│
├── data.py
├── geospatial_business_analysis.png
├── requirements.txt
├── README.md
```

---

## 📈 Business Value

This project enables:

✅ Identification of underserved high-demand regions  
✅ Data-driven retail expansion strategy  
✅ Market coverage optimization  
✅ Revenue maximization  
✅ Cluster-based store performance insights  

---

## 🔮 Future Enhancements

- Real-world census data integration  
- Interactive web dashboard (Streamlit / Plotly)  
- ROI forecasting model  
- Competitor location analysis  
- GIS shapefile integration  
- Real-time API-based demand data  

---

## 🎓 Academic / Portfolio Value

This project demonstrates:

- Geospatial analytics
- Business intelligence modeling
- Spatial clustering
- Data visualization
- Strategic decision analytics
- Machine learning integration

Suitable for:

- Data Science Portfolio
- Business Analytics Projects
- MBA Analytics Submission
- Retail Strategy Modeling
- Resume Showcase

---

## 👨‍💻 Author

Aryan  
Geospatial Business Intelligence Project  

---