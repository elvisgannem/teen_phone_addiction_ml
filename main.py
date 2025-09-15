import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/teen_phone_addiction_dataset.csv')

df = df.drop(columns=['ID', 'Location', 'Name'])

casos = df[
    (df['Daily_Usage_Hours'] == 0) &
    (
        (df['Time_on_Social_Media'] != 0) |
        (df['Time_on_Gaming'] != 0) |
        (df['Time_on_Education'] != 0)
    )
]

df.loc[df['Daily_Usage_Hours'] == 0, ['Time_on_Social_Media', 'Time_on_Gaming', 'Time_on_Education']] = 0

df['Gender'] = df['Gender'].astype('category').cat.codes
#0 - Female
#1 - Male
#2 - Other
df['Phone_Usage_Purpose'] = df['Phone_Usage_Purpose'].astype('category').cat.codes
#0 - Browsing
#1 - Education
#2 - Gaming
#3 - Other
#4 - Social Media

df['School_Grade'] = df['School_Grade'].astype(str).str.replace('th', '', regex=False)
df['School_Grade'] = pd.to_numeric(df['School_Grade'], errors='coerce')

colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
colunas_numericas.remove("Addiction_Level")

outliers = {}
for col in colunas_numericas + ['School_Grade']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
    outliers_count = outlier_mask.sum()
    if outliers_count > 0:
        outliers[col] = outliers_count

if outliers:
    print("Foram encontrados outliers nas seguintes colunas:")
    for col, count in outliers.items():
        print(f"{col}: {count} outliers")
    # Salvar boxplots para as colunas com outliers
    plt.figure(figsize=(12, 6))
    df[list(outliers.keys())].boxplot()
    plt.title("Boxplot das colunas com outliers detectados")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outliers_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo como 'outliers_boxplot.png'")
else:
    print("Não foram encontrados outliers nas colunas numéricas.")

df['Percent_Social_Media_Time'] = df['Time_on_Social_Media'] / df['Daily_Usage_Hours']
df['Percent_Gaming_Time'] = df['Time_on_Gaming'] / df['Daily_Usage_Hours']
df['Percent_Education_Time'] = df['Time_on_Education'] / df['Daily_Usage_Hours']
df['Sleep_to_Usage_Ratio'] = df['Sleep_Hours'] / df['Daily_Usage_Hours']
df['Usage_Intensity'] = df['Phone_Checks_Per_Day'] / df['Daily_Usage_Hours']


scaler = StandardScaler()
df[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])

X = df.drop(columns=["Addiction_Level"])
y = df["Addiction_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Shape treino:", X_train.shape, y_train.shape)
print("Shape teste:", X_test.shape, y_test.shape)





