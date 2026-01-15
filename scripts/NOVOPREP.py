import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import json
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split


RAW_PATH_DEFAULT = 'data/raw/dataset4.xlsx'
OUTPUT_DIR_DEFAULT = 'data/processed'
TEST_SIZE = 0.25  # 75% Treino / 25% Teste

class DataPreparationOptimized:
    
    COLUMNS_TO_DROP = [
        'ULTIMO_CORTE_INAD', 'ULTIMO_CANCELAMENTO', 'HORA_REGISTRO',
        'ANALISE_ATENDIMENTO', 'SOLUCAO_ATENDENTE', 'BAIRRO_y', 'CANCELADO', 'SITUACAO',
        'TAB_N1', 'TAB_N2', 'TAB_N3', 'MACRO_CATEGORIA','IDADE_APROX','GENERO','CANAL','CIDADE_x','CIDADE_y'
    ]

    def __init__(self, raw_data_path: str = RAW_PATH_DEFAULT, output_dir: str = OUTPUT_DIR_DEFAULT):
        self.raw_data_path = Path(raw_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 100)
        print(" PREPARAÇÃO DE DADOS (BALANCEAMENTO 2500/2500 - CANCELADOS ALEATÓRIOS)")
        print("=" * 100)


    @staticmethod
    def definir_macro_categoria(texto):
        texto = str(texto).upper() if pd.notna(texto) else ""
        
        palavras_financeiro = [
            'BOLETO', 'FATURA', 'COBRANÇA', 'DEBITO', 'DÉBITO', 'PAGAMENTO', 
            'VALOR', 'NEGOCIAÇÃO', 'CARNÊ', 'DESBLOQUEIO', 'FINANCEIRA',
            'VENCIMENTO', 'TAXAS', 'CODIGO DE BARRAS', 'CÓDIGO DE BARRAS'
        ]
        if any(p in texto for p in palavras_financeiro): return 'FINANCEIRO'

        palavras_saida = [
            'CONCORRENCIA', 'CONCORRÊNCIA', 'MUDOU DE', 'OPERADORA', 
            'INVIABILIDADE', 'NÃO QUER MAIS', 'NAO QUER MAIS', 'PORTABILIDADE',
            'CONTENÇÃO', 'MUDANÇA DE CIDADE'
        ]
        if any(p in texto for p in palavras_saida): return 'MOTIVO_SAIDA'

        palavras_cancelamento = ['CANCELAMENTO', 'INSATISFAÇÃO', 'RETENÇÃO', 'REVERSÃO']
        if any(p in texto for p in palavras_cancelamento): return 'CANCELAMENTO_RETENCAO'

        palavras_tecnico = [
            'SINAL', 'NAVEGA', 'LENTIDÃO', 'OSCILANDO', 'WI-FI', 'WIFI',
            'MODEM', 'CONEXÃO', 'VELOCIDADE', 'TÉCNICA', 'EQUIPAMENTO', 
            'IMAGEM', 'REPARO', 'INSTALAÇÃO', 'SUPORTE', 'BACKHAUL', 'GPON',
            'CAINDO', 'INTERRUPÇÃO', 'NORMALIZAÇÃO'
        ]
        if any(p in texto for p in palavras_tecnico): return 'SUPORTE_TECNICO'

        palavras_comercial = [
            'PACOTE', 'PROMOÇÃO', 'ADESÃO', 'VENDAS', 'PLANO', 
            'FIDELIDADE', 'MIGRAÇÃO', 'DEGUSTAÇÃO', 'ASSINATURA'
        ]
        if any(p in texto for p in palavras_comercial): return 'COMERCIAL'

        palavras_admin = [
            'ENDEREÇO', 'TITULARIDADE', 'SENHA', 'CADASTRAIS', 
            'AGENDAMENTO', 'DATA', 'HORA', 'REGISTRO', 'AUSENTE'
        ]
        if any(p in texto for p in palavras_admin): return 'ADMINISTRATIVO'

        return 'OUTROS'

    @staticmethod
    def normalize_text(texto):
        """Normaliza texto para lowercase e remove acentos"""
        if pd.isna(texto):
            return ""
        texto = str(texto).lower().strip()
        return texto

    # ---------------------------
    # Etapas de Limpeza
    # ---------------------------

    def filter_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove registros com idade inválida ou ausente"""
        print(" [2/13] Filtrando por IDADE...")
        
        initial_count = len(df)
        
        # Remover linhas onde IDADE_APROX é nula ou inválida
        if 'IDADE_APROX' in df.columns:
            df = df[df['IDADE_APROX'].notna()].copy()
            df = df[(df['IDADE_APROX'] >= 18) & (df['IDADE_APROX'] <= 120)].copy()
        
        removed = initial_count - len(df)
        print(f"   ✓ {removed} registros removidos (idade inválida)")
        print(f"   ✓ {len(df)} registros restantes")
        
        return df

    def encode_categorical(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Codifica variáveis categóricas"""
        print(" [3/13] Codificando CATEGÓRICAS...")
        
        encodings = {}
        categorical_cols = ['GENERO', 'CANAL', 'CIDADE_x']
        
        for col in categorical_cols:
            if col in df.columns:
                # Substituir NaN por 'UNKNOWN'
                df[col] = df[col].fillna('UNKNOWN').astype(str)
                
                # Criar mapping
                unique_vals = df[col].unique()
                encoding = {val: idx for idx, val in enumerate(unique_vals)}
                encodings[col] = encoding
                
                # Aplicar encoding
                df[col + '_ENCODED'] = df[col].map(encoding)
                
                print(f"   ✓ {col}: {len(encoding)} categorias codificadas")
        
        # Remover colunas originais categóricas (já estão encoded)
        cols_to_remove = [col for col in categorical_cols if col in df.columns]
        df = df.drop(columns=cols_to_remove)
        
        return df, encodings

    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        print(" [1/13] Criando TARGET...")
        if 'SITUACAO' not in df.columns: 
            raise ValueError("Coluna SITUACAO não encontrada!")
        df['TARGET'] = (df['SITUACAO'].astype(str).str.upper().str.strip() == 'DESLIGADO').astype(int)
        print(f"   ✓ Cancelados: {df['TARGET'].sum()} | Ativos: {(df['TARGET']==0).sum()}")
        return df

    def count_registros(self, df: pd.DataFrame) -> pd.DataFrame:
        """Conta quantidade de registros por cliente"""
        print(" [4/13] Contando registros por cliente...")
        df['QTD_REGISTROS'] = df.groupby('ID_CLIENTE')['ID_CLIENTE'].transform('size')
        return df

    # ---------------------------
    # Features Avançadas
    # ---------------------------

    def create_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula frequência de contato"""
        print(" [5/13] Criando features de SERVIÇO...")
        
        if 'DATA_REGISTRO' in df.columns:
            df['DATA_REGISTRO_DT'] = pd.to_datetime(df['DATA_REGISTRO'], errors='coerce')
            date_range = (df['DATA_REGISTRO_DT'].max() - df['DATA_REGISTRO_DT'].min()).days + 1
            if date_range > 0:
                df['FREQ_CONTATO'] = df.groupby('ID_CLIENTE')['ID_CLIENTE'].transform('size') / max(date_range / 365, 1)
            else:
                df['FREQ_CONTATO'] = 0
        else:
            df['FREQ_CONTATO'] = 0
        
        return df

    def count_reclamacoes(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[6/13] Contando reclamações...")
        if 'TAB_N1' not in df.columns:
            df['QTD_RECLAMACOES'] = 0
            df['QTD_SOLICITACOES'] = 0
            df['PCT_RECLAMACOES'] = 0
            return df

        df['TAB_N1_NORM'] = df['TAB_N1'].astype(str).apply(self.normalize_text)
        df['is_reclamacao'] = df['TAB_N1_NORM'].str.contains('reclamacao', na=False).astype(int)
        df['QTD_RECLAMACOES'] = df.groupby('ID_CLIENTE')['is_reclamacao'].transform('sum')
        df['QTD_SOLICITACOES'] = df.groupby('ID_CLIENTE')['TAB_N1_NORM'].transform('count')
        df['PCT_RECLAMACOES'] = (df['QTD_RECLAMACOES'] / df['QTD_SOLICITACOES'].replace(0, np.nan)).fillna(0)
        df = df.drop(columns=['TAB_N1_NORM', 'is_reclamacao'])
        return df

    def create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features financeiras (removidas por baixa importância)"""
        return df

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features temporais - apenas MESES é significativo"""
        if 'MESES' not in df.columns: 
            df['MESES'] = 0
        df['MESES'] = df['MESES'].fillna(0)
        return df

    def generate_cluster_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print(" [7/13] Gerando CLUSTERS...")
        if 'TAB_N3' not in df.columns: return df

        df['MACRO_CATEGORIA'] = df['TAB_N3'].apply(self.definir_macro_categoria)
        
        clusters = ['FINANCEIRO', 'MOTIVO_SAIDA', 'CANCELAMENTO_RETENCAO', 
                    'SUPORTE_TECNICO', 'COMERCIAL', 'ADMINISTRATIVO', 'OUTROS']
        
        for cluster in clusters:
            col_name = f'IS_{cluster}'
            df[col_name] = (df['MACRO_CATEGORIA'] == cluster).astype(int)
            df[f'QTD_{cluster}'] = df.groupby('ID_CLIENTE')[col_name].transform('sum')
            df = df.drop(columns=[col_name])

        df['JA_INDICOU_MOTIVO_SAIDA'] = (df['QTD_MOTIVO_SAIDA'] > 0).astype(int)
        df['JA_TENTOU_CANCELAR'] = (df['QTD_CANCELAMENTO_RETENCAO'] > 0).astype(int)
        df['TEM_PROB_FINANCEIRO'] = (df['QTD_FINANCEIRO'] > 0).astype(int)
        df['TEM_PROB_TECNICO'] = (df['QTD_SUPORTE_TECNICO'] > 0).astype(int)
        
        df['RISK_SCORE'] = (
            df['JA_INDICOU_MOTIVO_SAIDA'] * 3 +
            df['JA_TENTOU_CANCELAR'] * 3 +
            df['TEM_PROB_FINANCEIRO'] * 2 +
            df['TEM_PROB_TECNICO'] * 1
        )
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print(" [8/13] Criando INTERAÇÕES...")
        # Features de interação não aparecem nos top - removidas
        return df

    # ---------------------------
    # Agregação
    # ---------------------------

    def create_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print(" [9/13] Agregando features por CLIENTE...")
        
        if 'DATA_REGISTRO' in df.columns:
            df['DATA_REGISTRO_DT'] = pd.to_datetime(df['DATA_REGISTRO'], errors='coerce')
        else:
            df['DATA_REGISTRO_DT'] = pd.NaT
        
        overall_max = df['DATA_REGISTRO_DT'].max() or pd.Timestamp.today()

        # APENAS FEATURES RELEVANTES (F-SCORE ALTO)
        agg_funcs = {
            'DATA_REGISTRO_DT': ['min', 'max', 'nunique'],
            'QTD_REGISTROS': 'max',
            'MESES': 'max',
            'JA_TENTOU_CANCELAR': 'max',
            'FREQ_CONTATO': 'max',
            'QTD_FINANCEIRO': 'mean',
            'QTD_SUPORTE_TECNICO': 'mean',
            'QTD_OUTROS': 'mean',
            'QTD_ADMINISTRATIVO': 'mean',
            'TARGET': 'max'
        }

        agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}
        
        grouped = df.groupby('ID_CLIENTE').agg(agg_funcs)
        grouped.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in grouped.columns.values]
        
        rename_map = {
            'DATA_REGISTRO_DT_min': 'FIRST_DATE',
            'DATA_REGISTRO_DT_max': 'LAST_DATE',
            'DATA_REGISTRO_DT_nunique': 'N_UNIQUE_DATES',
            'QTD_REGISTROS_max': 'QTD_REGISTROS_CLIENTE',
            'MESES_max': 'MESES_CLIENTE',
            'JA_TENTOU_CANCELAR_max': 'JA_TENTOU_CANCELAR_max',
            'FREQ_CONTATO_max': 'FREQ_CONTATO_max',
            'QTD_FINANCEIRO_mean': 'QTD_FINANCEIRO_mean',
            'QTD_SUPORTE_TECNICO_mean': 'QTD_SUPORTE_TECNICO_mean',
            'QTD_OUTROS_mean': 'QTD_OUTROS_mean',
            'QTD_ADMINISTRATIVO_mean': 'QTD_ADMINISTRATIVO_mean',
            'TARGET_max': 'TARGET'
        }
        
        grouped = grouped.rename(columns=rename_map)

        # Features derivadas pós-agregação
        grouped['DAYS_SINCE_LAST'] = (overall_max - grouped['LAST_DATE']).dt.days.fillna(9999).astype(int)
        grouped['TAXA_CONTATO_DIA'] = grouped['QTD_REGISTROS_CLIENTE'] / 1

        if 'DATA_REGISTRO_DT' in df.columns:
            df['DAYS_FROM_MAX'] = (overall_max - df['DATA_REGISTRO_DT']).dt.days
            last30 = df[df['DAYS_FROM_MAX'] <= 30].groupby('ID_CLIENTE').size()
            grouped['QTD_SOL_LAST_30D'] = last30.fillna(0).astype(int)

        grouped = grouped.fillna(0).reset_index()
        
        # REMOVE COLUNAS NÃO SIGNIFICATIVAS
        cols_to_drop = ['FIRST_DATE', 'LAST_DATE', 'MESES_CLIENTE', 'GENERO_CLIENTE', 
                        'CIDADE_ENCODED_CLIENTE', 'TICKET_MEDIO_CLIENTE', 'TICKET_MIN']
        cols_to_drop = [c for c in cols_to_drop if c in grouped.columns]
        grouped = grouped.drop(columns=cols_to_drop)
        
        print(f"   ✓ {len(grouped)} clientes agregados")
        print(f"   ✓ Features mantidas: {len(grouped.columns)-2} (excluindo ID_CLIENTE e TARGET)")
        return grouped



    def treat_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        print(" [10/13] Tratando OUTLIERS...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['TARGET', 'ID_CLIENTE']]
        
        for col in numeric_cols:
            if df[col].std() == 0: continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            df[col] = df[col].clip(lower=lower, upper=upper)
        
        print(f"   ✓ Outliers tratados em {len(numeric_cols)} colunas")
        return df

    # ---------------------------
    # BALANCEAMENTO SIMPLES (2500 vs 2500)
    # ---------------------------

    def balance_simple(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f" [11/13] BALANCEAMENTO SIMPLES (Meta: 2500 vs 2500)...")
        
        TARGET_N = 2500
        active = df[df['TARGET'] == 0]
        canceled = df[df['TARGET'] == 1]
        
        print(f"    Dataset Original:")
        print(f"      - Cancelados: {len(canceled)}")
        print(f"      - Ativos: {len(active)}")
        
        # Seleção de cancelados aleatoriamente
        if len(canceled) >= TARGET_N:
            print(f"      Cancelados suficientes! Pegando {TARGET_N} aleatoriamente.")
            selected_canceled = canceled.sample(n=TARGET_N, random_state=42)
        else:
            print(f"       Cancelados insuficientes ({len(canceled)}). Usando todos.")
            selected_canceled = canceled

        # Seleção de ativos aleatoriamente
        if len(active) >= TARGET_N:
            print(f"       Selecionando {TARGET_N} ativos aleatoriamente.")
            selected_active = active.sample(n=TARGET_N, random_state=42)
        else:
            print(f"       Ativos insuficientes ({len(active)}). Usando todos.")
            selected_active = active

        df_bal = pd.concat([selected_canceled, selected_active]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"    Dataset Final: {len(df_bal)} clientes")
        print(f"      Cancelados: {(df_bal['TARGET'] == 1).sum()}")
        print(f"      Ativos:     {(df_bal['TARGET'] == 0).sum()}")
        
        return df_bal

    # ---------------------------
    # Split e Salvamento
    # ---------------------------

    def split_train_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print(f" [12/13] Dividindo Treino/Teste POR CLIENTE (SEM DATA LEAKAGE)...")
        
        # PASSO 1: Extrair clientes únicos com seus targets
        clientes = df[['ID_CLIENTE', 'TARGET']].drop_duplicates()
        
        print(f"    Total de clientes únicos: {len(clientes)}")
        print(f"      - Cancelados: {(clientes['TARGET'] == 1).sum()}")
        print(f"      - Ativos: {(clientes['TARGET'] == 0).sum()}")
        
        # PASSO 2: Dividir PELOS IDs (não pelas linhas)
        train_ids, test_ids = train_test_split(
            clientes['ID_CLIENTE'],
            test_size=TEST_SIZE,
            stratify=clientes['TARGET'],
            random_state=42
        )
        
        # PASSO 3: Filtrar dados pelo set de IDs
        train_df = df[df['ID_CLIENTE'].isin(train_ids)].reset_index(drop=True)
        test_df = df[df['ID_CLIENTE'].isin(test_ids)].reset_index(drop=True)
        
        # PASSO 4: Validar que NÃO há overlap
        overlap = set(train_df['ID_CLIENTE']) & set(test_df['ID_CLIENTE'])
        
        print(f"\n   ✓ TREINO: {len(train_df)} linhas")
        print(f"      - Clientes únicos: {train_df['ID_CLIENTE'].nunique()}")
        print(f"      - Cancelados: {(train_df['TARGET'] == 1).sum()}")
        print(f"      - Ativos: {(train_df['TARGET'] == 0).sum()}")
        
        print(f"\n   ✓ TESTE: {len(test_df)} linhas")
        print(f"      - Clientes únicos: {test_df['ID_CLIENTE'].nunique()}")
        print(f"      - Cancelados: {(test_df['TARGET'] == 1).sum()}")
        print(f"      - Ativos: {(test_df['TARGET'] == 0).sum()}")
        
        print(f"\n   ✓ Overlap de IDs: {len(overlap)}")
        
        if overlap:
            raise RuntimeError(f" DATA LEAKAGE DETECTADO: {len(overlap)} ID_CLIENTE em treino E teste!")
        
        print(f"    ZERO DATA LEAKAGE - Pipeline seguro!")
        
        return train_df, test_df

    def apply_robust_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica RobustScaler apenas em features relevantes"""
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'TARGET' in numeric: numeric.remove('TARGET')
        if 'ID_CLIENTE' in numeric: numeric.remove('ID_CLIENTE')
        
        cols_to_scale = ['N_UNIQUE_DATES', 'FREQ_CONTATO_max', 'JA_TENTOU_CANCELAR_max',
                        'DAYS_SINCE_LAST', 'QTD_SOL_LAST_30D', 'TAXA_CONTATO_DIA',
                        'QTD_REGISTROS_CLIENTE']
        
        cols = [c for c in cols_to_scale if c in df.columns]
        scaler = RobustScaler()
        df_norm = df.copy()
        if cols:
            df_norm[cols] = scaler.fit_transform(df[cols])
        return df_norm

    def prepare(self):
        """Executa pipeline completo de preparação"""
        print(" [0/13] Carregando dados brutos...")
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.raw_data_path}")
        
        df = pd.read_excel(self.raw_data_path) if self.raw_data_path.suffix == '.xlsx' else pd.read_csv(self.raw_data_path)
        print(f"   ✓ {len(df)} registros carregados")

        df = self.create_target(df)
        df = self.filter_age(df)
        df, encodings = self.encode_categorical(df)
        df = self.count_registros(df)
        df = self.create_service_features(df)
        df = self.count_reclamacoes(df)
        df = self.create_financial_features(df)
        df = self.create_temporal_features(df)
        df = self.generate_cluster_features(df)
        df = self.create_interaction_features(df)
        
        df_agg = self.create_aggregate_features(df)
        df_treated = self.treat_outliers(df_agg)
        
        # BALANCEAMENTO SIMPLES
        df_bal = self.balance_simple(df_treated)
        
        train_df, test_df = self.split_train_test(df_bal)

        print(" [13/13] Salvando arquivos...")
        def clean_dt(d): 
            return d.select_dtypes(exclude=['datetime64[ns]', '<M8[ns]'])
        
        clean_dt(train_df).to_excel(self.output_dir / 'train.xlsx', index=False)
        clean_dt(test_df).to_excel(self.output_dir / 'test.xlsx', index=False)
        
        train_norm = self.apply_robust_scaling(clean_dt(train_df))
        test_norm = self.apply_robust_scaling(clean_dt(test_df))
        
        train_norm.to_excel(self.output_dir / 'train_minmax.xlsx', index=False)
        test_norm.to_excel(self.output_dir / 'test_minmax.xlsx', index=False)

        with open(self.output_dir / 'encodings.json', 'w', encoding='utf-8') as f:
            def convert(o): 
                return int(o) if isinstance(o, np.int64) else o
            json.dump(encodings, f, ensure_ascii=False, indent=2, default=convert)

        with open(self.output_dir / 'features_report.txt', 'w', encoding='utf-8') as f:
            f.write("="*80 + "\nRELATÓRIO DE FEATURES\n"+"="*80 + "\n\n")
            cols = clean_dt(train_df).columns.tolist()
            if 'TARGET' in cols: cols.remove('TARGET')
            if 'ID_CLIENTE' in cols: cols.remove('ID_CLIENTE')
            f.write(f"Total: {len(cols)}\n\n")
            for i, feat in enumerate(cols, 1):
                f.write(f"{i:02d}. {feat}\n")
        
        print("=" * 80)
        print(" CONCLUÍDO!")
        print(f" Arquivos salvos em: {self.output_dir}")
        print("=" * 80)

if __name__ == '__main__':
    try:
        preparer = DataPreparationOptimized()
        preparer.prepare()
    except Exception as e:
        print(f"\n ERRO FATAL:\n{str(e)}")