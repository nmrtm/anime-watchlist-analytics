# MyAnimeWatch: End-to-End Analytics & Recommendations

## 🎴 **Project Overview**  
This project is a comprehensive, end-to-end analytics pipeline built on my personal anime watch history. It combines custom data collection, cleaning, and exploratory analysis to uncover patterns in my viewing preferences across anime franchises and their genres, themes, and demographics. The ultimate goal is to generate personalised recommendations that balance familiarity and novelty - offering smarter suggestions for what to watch next. This project also lays the foundation for a future personalised anime recommendation system.

---

## 🏢 **Project Context**  
Many existing anime datasets are outdated or incomplete, and generic “top anime” lists often fail to recommend shows that truly match individual tastes. Starting with my own watchlist, including sequels, prequels, spin-offs, and specials, this project fills the gap by creating a personal dataset using the Jikan API (an unofficial MyAnimeList API). It goes beyond individual titles, focusing on entire franchises including prequels, OVAs, TV specials etc., and their metadata.

---

## 🎯 **Objectives**
- Collect detailed, up-to-date metadata for all anime in my personal watchlist and their related franchises.  
- Clean, preprocess, and aggregate data to enable meaningful analysis.  
- Identify trends and underexplored areas in my anime preferences (genres, themes, demographics).  
- Use insights to guide ongoing personalised recommendations, improving the discovery of new shows aligned with my taste profile.

---

## 📊 **Process Overview**

### Phase 1: Data Collection (`phase1_collection.py`)  
- Input: Personal watchlist of completed anime.  
- Data fetched via Jikan API with handling for rate limits and retries.  
- Standardised titles using fuzzy matching and collected franchise-level data, including related shows.  
- Processed data in batches and completed episode counts.  
- Output: A comprehensive CSV dataset combining all watched franchises.

### Phase 2: Data Cleaning & Preprocessing (`phase2_cleaning.ipynb`) 
- Removed duplicates, irrelevant or unaired entries.  
- Normalised fields, engineered features, and consolidated related titles by franchise.  
- Addressed missing demographic data using franchise-based backfilling.  
- Applied Bayesian shrinkage to stabilise scores, reducing bias from low-rating volumes.  
- Output: Cleaned & aggregated dataset ready for analysis.

### Phase 3: Visualisation & Insights (`phase3_viz.ipynb`) 
- Analysed top and “second-most” watched genres, themes, and demographics.  
- Identified balanced “sweet spots” for recommendation—categories I enjoy but haven’t saturated.  
- Examined trends by year, popularity, and score, revealing personal viewing history insights.  
- Visualisations and interactive dashboards complement findings.

---

## 🔍 **Key Insights**
- “Next five” genres/themes offer a fresh but familiar shortlist for what to watch next.
- **Genres:** Action and Adventure dominate; secondary favorites include Mystery, Supernatural, and Sci-Fi.  
- **Themes:** Preference for mature, school, and intense narratives, with potential areas to explore like Parody and Organized Crime.  
- **Demographics:** Centered on Shonen, with notable interest in Seinen and Shojo categories.  
- Viewing spans from the nostalgic “golden age” of 2005–2010, which forms a core era in my preferences, while newer shows from 2020 onward are also prevalent.
- Score vs. popularity chart highlight hidden gems beyond mainstream hits.

---

## 🚧 **Limitations**
- **Personal Dataset Bias:** All insights are based on my own watch history (50+ titles); results are not representative of broader anime trends or audiences.
- **API & Data Instability:** The Jikan API (used for scraping MyAnimeList) can be slow, unstable, or return incomplete/missing fields.
- **Imperfect Franchise Grouping:** Related anime (sequels, spin-offs, etc.) may sometimes be missed or wrongly grouped due to inconsistent MyAnimeList franchise links.
- **Demographic & Theme Gaps:** ~30% of titles have missing demographic labels; theme tags are inconsistently applied, limiting the precision of those insights.
- **Loss of Granularity:** Aggregating at the franchise level hides differences between individual series, OVAs, and specials within the same franchise.
---

## 💡 **Future Scope**  
- **Batch Processing & Faster Collection:** Save progress in batches and experiment with parallel API calls to speed up data collection for larger watchlists.
- **User Watch History Integration:** Use actual watch logs to differentiate watched, planned, and dropped anime for more precise insights.
- **Enhanced Data Coverage:** Integrate more comprehensive datasets (Kaggle, AniList, etc.) to fill gaps in demographics and themes.
- **Advanced Analysis:** Use text embeddings from anime synopses and review sentiment analysis for richer insights and smarter recommendations.
- **Full Recommender System:** Develop a personalised recommender using weighted genre, theme, and demographic similarities.
   
*Read my proposal of the anime recommender on [Notion](https://www.notion.so/MyAnimeWatch-End-to-End-Analytics-Recommendations-23ba1fc6e1a380789c88cf3fb8269f34?source=copy_link#287a1fc6e1a380af9201d36794388fa2).*

---

## 📁 **Files in this Repo**  
- `README.md`: Project overview and documentation.  
- `phase1_collection.py`: Data collection scripts for fetching and batch processing anime metadata via Jikan API.  
- `phase2_cleaning.ipynb`: Notebook for cleaning, filtering, and aggregating raw data.  
- `phase3_viz.ipynb`: Notebook for exploratory data analysis and visualisations.  

---

## 🛠️ **Tools Used**  
- Python 3.11+  
- Jikan API (MyAnimeList unofficial API)  
- Jupyter Notebook for analysis and visualisation  

---

## 🔗 **Explore More**  
- Full data collection and cleaning walkthroughs available on [Notion](https://www.notion.so/MyAnimeWatch-End-to-End-Analytics-Recommendations-23ba1fc6e1a380789c88cf3fb8269f34)  

---

## 👩🏽‍💻 **About Me**

[View my Portfolio](https://www.notion.so/Namrata-s-Data-Corner-1fea1fc6e1a380feb078df49d0bb5bc6)  

[Connect with me on LinkedIn](https://www.linkedin.com/in/namratamuralidharan/) 

*If you use or adapt this project, please credit or connect - I’d love to see your take on it!*

