import pandas as pd
import os

RAW_DIR = 'data/raw'
OUTPUT_DIR = 'data/processed'

os.makedirs(OUTPUT_DIR, exist_ok=True)

df23 = pd.read_excel(os.path.join(RAW_DIR, "registro_atendimento_2023.xlsx"))
df24 = pd.read_excel(os.path.join(RAW_DIR, "registros_2024.xlsx"))
df25 = pd.read_excel(os.path.join(RAW_DIR, "registros_2025.xlsx"))
perfil = pd.read_excel(os.path.join(RAW_DIR, "perfil_clienteV2.xlsx"))

for df in [df23, df24, df25, perfil]:
    df["ID_CLIENTE"] = df["ID_CLIENTE"].astype(str).str.strip()

atendimentos = pd.concat([df23, df24, df25], ignore_index=True)

print(f"Total de atendimentos: {len(atendimentos)}")

df_final = atendimentos.merge(
    perfil,
    on="ID_CLIENTE",
    how="inner"   
)

output_path = os.path.join(OUTPUT_DIR, "base_consolidada_v1.xlsx")
df_final.to_excel(output_path, index=False)


print(f" Arquivo salvo em: {output_path}")
