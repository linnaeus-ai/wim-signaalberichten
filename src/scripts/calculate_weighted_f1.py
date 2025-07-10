import pandas as pd

def calculate_weighted_f1(df):
    records = df.to_dict('records')
    total_support = sum(label["tp"] + label["fn"] for label in records)
    weighted_f1 = sum((label["tp"] + label["fn"]) * label["f1_score"] for label in records) / total_support

    return weighted_f1

if __name__ == "__main__":
    df_onderwerp = pd.read_excel("src/data/Test 2 wel product-dienst.xlsx", sheet_name="onderwerp_scores")
    weighted_f1_onderwerp = calculate_weighted_f1(df_onderwerp)
    df_beleving = pd.read_excel("src/data/Test 2 wel product-dienst.xlsx", sheet_name="beleving_scores")
    weighted_f1_beleving = calculate_weighted_f1(df_beleving)
    df_total = pd.concat([df_onderwerp, df_beleving], ignore_index=True)
    weighted_f1_total = calculate_weighted_f1(df_total)
    
    print(f"Onderwerp weighted f1 score: {weighted_f1_onderwerp}")
    print(f"Beleving weighted f1 score: {weighted_f1_beleving}")
    print(f"Total weighted f1 score: {weighted_f1_total}")
