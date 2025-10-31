# app/eda.py
import pandas as pd
import numpy as np

'''def do_eda(session):
    df = session.get("df")
    if df is None:
        raise ValueError("No dataset in session.")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    missing_values = df.isnull().sum().to_dict()
    missing_percentage = (df.isnull().mean() * 100).round(2).to_dict()
    summary_stats = {}
    for c in numeric_columns:
        s = df[c].describe()
        summary_stats[c] = {
            "mean": float(s["mean"]) if pd.notna(s["mean"]) else None,
            "std": float(s["std"]) if pd.notna(s["std"]) else None,
            "min": float(s["min"]) if pd.notna(s["min"]) else None,
            "max": float(s["max"]) if pd.notna(s["max"]) else None,
        }
    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_values": {k: int(v) for k, v in missing_values.items()},
        "missing_percentage": missing_percentage,
        "summary_stats": summary_stats,
    }'''
import os
import seaborn as sns
import matplotlib.pyplot as plt

def do_bivariate_eda(session):
    """
    Perform Bivariate EDA and save plots to static/plots folder.
    """

    df = session.get("df")
    if df is None:
        raise ValueError("No dataset in session.")

    os.makedirs("static/plots", exist_ok=True)
    plot_dir = "static/plots"

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    print("\n🚀 Starting Bivariate EDA...\n")

    # === 1️⃣ Correlation Heatmap ===
    if len(num_cols) > 1:
        print("📊 Correlation Heatmap for Numerical Features")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation Between Numerical Columns")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/correlation_heatmap.png", bbox_inches="tight")
        plt.close()
    else:
        print("Not enough numerical columns for correlation analysis.\n")

    # === 2️⃣ Pairwise Scatterplots (first 3 numeric columns) ===
    if len(num_cols) >= 2:
        print("📈 Pairwise Scatterplots (first 3 numeric columns)")
        sns.pairplot(df[num_cols[:3]])
        plt.savefig(f"{plot_dir}/pairplot_numeric.png", bbox_inches="tight")
        plt.close()
    else:
        print("Not enough numeric columns for pairplot.\n")

    # === 3️⃣ Boxplots for Numeric vs Categorical ===
    if len(num_cols) > 0 and len(cat_cols) > 0:
        print("📦 Boxplots: Numerical vs Categorical")
        for cat in cat_cols[:2]:
            for num in num_cols[:2]:
                plt.figure(figsize=(7, 4))
                sns.boxplot(x=cat, y=num, data=df, palette='Set3')
                plt.title(f"{num} Distribution across {cat}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                file_name = f"{num}_vs_{cat}_box.png".replace(" ", "_")
                plt.savefig(f"{plot_dir}/{file_name}", bbox_inches="tight")
                plt.close()
    else:
        print("Not enough data for categorical-numerical analysis.\n")

    # === 4️⃣ Categorical–Categorical Relationship Heatmap ===
    if len(cat_cols) > 1:
        print("🧩 Categorical–Categorical Relationship Heatmap")
        cross_tab = pd.crosstab(df[cat_cols[0]], df[cat_cols[1]])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cross_tab, annot=True, cmap='YlGnBu')
        plt.title(f"Relationship between {cat_cols[0]} and {cat_cols[1]}")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/categorical_relationship.png", bbox_inches="tight")
        plt.close()
    else:
        print("Not enough categorical columns for cross-analysis.\n")

    print(f"\n✅ All Bivariate EDA plots saved in: {plot_dir}\n")
    session["eda_plots"] = os.listdir(plot_dir)
    return {"status": "success", "plots_saved_in": plot_dir, "plots": os.listdir(plot_dir)}


def do_validate(session):
    df = session.get("df")
    if df is None:
        raise ValueError("No dataset in session.")
    duplicates = int(df.duplicated().sum())
    total_missing = int(df.isnull().sum().sum())
    columns_with_missing = [c for c in df.columns if df[c].isnull().any()]
    return {"duplicates": duplicates, "total_missing": total_missing, "columns_with_missing": columns_with_missing}

def do_clean(session, strategy="mean"):
    df = session.get("df")
    if df is None:
        raise ValueError("No dataset in session.")
    cleaned_df = df.copy()
    if strategy == "drop":
        cleaned_df = cleaned_df.dropna()
    else:
        num_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            cleaned_df[c].fillna(cleaned_df[c].mean() if strategy == "mean" else cleaned_df[c].median(), inplace=True)
    session["df_clean"] = cleaned_df
    session["data"] = cleaned_df.to_dict(orient="records")
    return {"strategy_used": strategy, "new_shape": cleaned_df.shape}
