# Spotify Music Analysis (2015–2025)

An exploratory data analysis project on a Spotify dataset containing ~85,000 tracks released between 2015 and 2025. The notebook performs data cleaning, outlier detection, correlation analysis, and dimensionality reduction to understand what drives track popularity.

---

## Dataset

**Source:** Kaggle (Spotify 2015–2025 dataset)  
**Size:** 85,000 rows × 19 columns  
**Time span:** 2015–2025

### Features

| Column | Type | Description |
|--------|------|-------------|
| `track_id` | object | Unique track identifier |
| `track_name` | object | Song title |
| `artist_name` | object | Performing artist |
| `album_name` | object | Album title |
| `release_date` | object | Release date (YYYY-MM-DD) |
| `genre` | object | Music genre (Pop, Rock, Hip-Hop, EDM, etc.) |
| `duration_ms` | int64 | Track length in milliseconds |
| `popularity` | int64 | Spotify popularity score (0–100) |
| `danceability` | float64 | How suitable for dancing (0–1) |
| `energy` | float64 | Perceptual intensity (0–1) |
| `key` | int64 | Estimated key (0–11) |
| `loudness` | float64 | Overall loudness in dB |
| `mode` | int64 | Major (1) or minor (0) |
| `instrumentalness` | float64 | Likelihood of no vocals (0–1) |
| `tempo` | float64 | Beats per minute |
| `stream_count` | int64 | Total streams |
| `country` | object | Country of origin |
| `explicit` | int64 | Explicit content flag (0/1) |
| `label` | object | Record label |

---

## Analysis Pipeline

### 1. Data Cleaning
- **Missing values:** 21 missing `track_name`, 46 missing `album_name` → dropped rows with `dropna()`
- **Duplicates:** Checked and removed (no duplicates found after cleaning)
- **Final shape:** 84,933 rows × 19 columns

### 2. Outlier Detection (IQR Method)
Applied Tukey's IQR method on numerical features:
- `duration_ms`, `popularity`, `danceability`, `energy`, `loudness`, `tempo`, `instrumentalness`, `stream_count`
- Outliers capped at `Q1 - 1.5×IQR` and `Q3 + 1.5×IQR`
- Verified with before/after boxplots

### 3. Correlation Analysis
Computed correlation matrix between audio features and `popularity`:
- **Strongest positive correlations:** `energy`, `danceability`, `loudness`
- **Negative correlation:** `instrumentalness` (more instrumental → less popular)
- Visualized as annotated heatmap

### 4. Principal Component Analysis (PCA)
- Features: `danceability`, `energy`, `loudness`, `tempo`, `instrumentalness`, `duration_ms`
- Standardized with `StandardScaler`
- **2 components explain ~33.7% of variance** (PC1: 16.9%, PC2: 16.8%)
- Scatter plot colored by popularity shows clustering patterns

---

## Key Findings

- **Energy** and **danceability** are the strongest predictors of popularity
- **Instrumentalness** negatively correlates with popularity (vocal tracks tend to be more popular)
- **Loudness** shows a positive correlation (louder tracks → more popular)
- **Tempo** and **duration** have weaker linear relationships with popularity
- PCA reveals that the first two components separate tracks by energy/danceability vs. instrumental/acoustic qualities

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

## Usage

```bash
jupyter notebook project.ipynb
```

Run cells sequentially. The notebook is self-contained and will:
1. Load the CSV (update the path if needed)
2. Clean the data
3. Detect and handle outliers
4. Generate correlation heatmap
5. Perform PCA and visualize results

---

## File Structure

```
.
├── project.ipynb              # Main analysis notebook
├── spotify_2015_2025_85k.csv  # Dataset (85k tracks)
```

---

## Future Work

- Build a regression model to predict popularity from audio features
- Time-series analysis of genre trends over 2015–2025
- Country-level analysis of listening preferences
- Label/artist-level popularity drivers
- Cluster analysis to identify track archetypes