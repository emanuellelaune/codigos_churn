from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import joblib
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, accuracy_score, roc_curve, auc,
    classification_report
)

class GradientBoostingTrainerCustom:
    """
    Trainer customizado para receber datasets prontos.
    Gera ANOVA, correlações e relatório completo.
    """
    
    def __init__(
        self,
        train_path: str,
        test_path: str,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        max_depth: int = 8,
        subsample: float = 0.8,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        random_state: int = 42,
        output_dir: str = "outputs"
    ):
        """
        Args:
            train_path: Caminho para train.xlsx (com TARGET)
            test_path: Caminho para test.xlsx (com TARGET)
            n_estimators: Número de árvores (aumentado para melhor generalização)
            learning_rate: Taxa de aprendizado (reduzida para evitar overfitting)
            max_depth: Profundidade máxima (reduzida para árvores mais simples)
            subsample: Fração de amostras para treinar cada árvore (reduz overfitting)
            min_samples_split: Mínimo de amostras para dividir nó (previne overfitting)
            min_samples_leaf: Mínimo de amostras em folha (previne overfitting)
            random_state: Seed
            output_dir: Diretório para salvar outputs
        """
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        self.model: Optional[GradientBoostingClassifier] = None
        self.feature_names: list = []
        self.best_threshold: float = 0.5
        
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carrega datasets de arquivos Excel"""
        print("Carregando datasets...")
        
        df_train = pd.read_excel(self.train_path)
        df_test = pd.read_excel(self.test_path)
        
        print(f"  Train: {df_train.shape[0]} linhas, {df_train.shape[1]} colunas")
        print(f"  Test:  {df_test.shape[0]} linhas, {df_test.shape[1]} colunas")
        
        return df_train, df_test
    
    def prepare_features(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        """Prepara features e target"""
        print(" Preparando features...")
        
        
        if 'TARGET' not in df_train.columns:
            raise ValueError("Coluna TARGET não encontrada no dataset de treino!")
        if 'TARGET' not in df_test.columns:
            raise ValueError("Coluna TARGET não encontrada no dataset de teste!")
        
        self.y_train = df_train['TARGET'].fillna(0).astype(int)
        self.y_test = df_test['TARGET'].fillna(0).astype(int)
        
        
        cols_drop = ['TARGET', 'ID_CLIENTE', 'FIRST_DATE', 'LAST_DATE',]
        X_train = df_train.drop(columns=cols_drop, errors='ignore')
        X_test = df_test.drop(columns=cols_drop, errors='ignore')
        
        self.X_train = X_train.select_dtypes(include=[np.number]).fillna(0)
        self.X_test = X_test.select_dtypes(include=[np.number]).fillna(0)
        
        common_cols = list(set(self.X_train.columns) & set(self.X_test.columns))
        self.X_train = self.X_train[common_cols]
        self.X_test = self.X_test[common_cols]
        
        self.feature_names = self.X_train.columns.tolist()
        
        print(f"   ✓ Features numéricas: {len(self.feature_names)}")
        print(f"   ✓ Distribuição TARGET (train): {self.y_train.value_counts().to_dict()}")
        print(f"   ✓ Distribuição TARGET (test):  {self.y_test.value_counts().to_dict()}")
    
    def build_model(self) -> GradientBoostingClassifier:
        """Constrói modelo com regularização para evitar overfitting"""
        print(" Construindo Gradient Boosting (Anti-Overfitting)...")
        return GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            verbose=0
        )
    
    def fit_model(self):
        """Treina modelo com validação cruzada para monitorar overfitting"""
        print(" Treinando modelo...")
        self.model = self.build_model()
        self.model.fit(self.X_train, self.y_train)
        
        # Validação cruzada para verificar generalização
        print(" Validando com Cross-Validation (5-folds)...")
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        print(f"   ✓ CV Scores: {cv_scores}")
        print(f"   ✓ CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print("   ✓ Treinamento concluído")
    
    def calculate_anova(self) -> pd.DataFrame:
        """Calcula ANOVA F-test para cada feature vs TARGET"""
        print(" Calculando ANOVA...")
        
        anova_results = []
        
        for feature in self.feature_names:
            
            group_0 = self.X_train[self.y_train == 0][feature].values
            group_1 = self.X_train[self.y_train == 1][feature].values
            
            
            f_stat, p_value = stats.f_oneway(group_0, group_1)
            
            
            grand_mean = self.X_train[feature].mean()
            ss_between = len(group_0) * (group_0.mean() - grand_mean)**2 + \
                        len(group_1) * (group_1.mean() - grand_mean)**2
            ss_total = ((self.X_train[feature] - grand_mean)**2).sum()
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            significance = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "NS"))
            
            anova_results.append({
                'Feature': feature,
                'F_Statistic': f_stat,
                'P_Value': p_value,
                'Eta_Squared': eta_squared,
                'Significance': significance
            })
        
        anova_df = pd.DataFrame(anova_results).sort_values('F_Statistic', ascending=False).reset_index(drop=True)
        anova_df['Rank'] = range(1, len(anova_df) + 1)
        
        return anova_df[['Rank', 'Feature', 'F_Statistic', 'P_Value', 'Eta_Squared', 'Significance']]
    
    def calculate_correlations(self) -> pd.DataFrame:
        """Calcula correlação de Pearson entre features e TARGET"""
        print(" Calculando correlações...")
        
        X_with_target = self.X_train.copy()
        X_with_target['TARGET'] = self.y_train
        
        correlations = []
        for feature in self.feature_names:
            corr = X_with_target[feature].corr(X_with_target['TARGET'])
            correlations.append({
                'Feature': feature,
                'Correlation': corr,
                'Abs_Correlation': abs(corr)
            })
        
        corr_df = pd.DataFrame(correlations).sort_values('Abs_Correlation', ascending=False).reset_index(drop=True)
        corr_df['Rank'] = range(1, len(corr_df) + 1)
        
        return corr_df[['Rank', 'Feature', 'Correlation', 'Abs_Correlation']]
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, dataset_name: str = "Test") -> Dict:
        """Avalia modelo com métricas completas"""
        print(f" Avaliando {dataset_name}...")
        
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= self.best_threshold).astype(int)
        
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'Accuracy': accuracy_score(y, y_pred),
            'AUC_ROC': roc_auc_score(y, y_pred_proba),
            'Precision': precision_score(y, y_pred, zero_division=0),
            'Recall': recall_score(y, y_pred),
            'F1_Score': f1_score(y, y_pred),
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'True_Positives': tp,
            'True_Negatives': tn,
            'False_Positives': fp,
            'False_Negatives': fn,
            'Total_Samples': len(y),
            'Threshold': self.best_threshold
        }
        
        return metrics
    
    def save_metrics_report(self, metrics_train: Dict, metrics_test: Dict, filename: str = "metricas_completas.txt"):
        """Salva relatório completo de métricas em TXT"""
        print(f" Salvando relatório de métricas...")
        
        report_path = self.output_dir / filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 90 + "\n")
            f.write("RELATÓRIO COMPLETO DE MÉTRICAS - GRADIENT BOOSTING CHURN\n")
            f.write("=" * 90 + "\n\n")
            
            # Configuração do modelo
            f.write("CONFIGURAÇÃO DO MODELO\n")
            f.write("-" * 90 + "\n")
            f.write(f"N_Estimators:      {self.n_estimators}\n")
            f.write(f"Learning_Rate:     {self.learning_rate}\n")
            f.write(f"Max_Depth:         {self.max_depth}\n")
            f.write(f"Subsample:         {self.subsample}\n")
            f.write(f"Min_Samples_Split: {self.min_samples_split}\n")
            f.write(f"Min_Samples_Leaf:  {self.min_samples_leaf}\n")
            f.write(f"Threshold:         {self.best_threshold}\n")
            f.write(f"Random_State:      {self.random_state}\n\n")
            
            # Métricas de Treino
            f.write("MÉTRICAS DE TREINO\n")
            f.write("-" * 90 + "\n")
            f.write(f"Accuracy:      {metrics_train['Accuracy']:.6f}\n")
            f.write(f"AUC-ROC:       {metrics_train['AUC_ROC']:.6f}\n")
            f.write(f"Precision:     {metrics_train['Precision']:.6f}\n")
            f.write(f"Recall:        {metrics_train['Recall']:.6f}\n")
            f.write(f"F1-Score:      {metrics_train['F1_Score']:.6f}\n")
            f.write(f"Specificity:   {metrics_train['Specificity']:.6f}\n\n")
            
            f.write(f"Matriz de Confusão (Treino):\n")
            f.write(f"  True Negatives:  {metrics_train['True_Negatives']:,}\n")
            f.write(f"  False Positives: {metrics_train['False_Positives']:,}\n")
            f.write(f"  False Negatives: {metrics_train['False_Negatives']:,}\n")
            f.write(f"  True Positives:  {metrics_train['True_Positives']:,}\n\n")
            
            # Métricas de Teste
            f.write("MÉTRICAS DE TESTE\n")
            f.write("-" * 90 + "\n")
            f.write(f"Accuracy:      {metrics_test['Accuracy']:.6f}\n")
            f.write(f"AUC-ROC:       {metrics_test['AUC_ROC']:.6f}\n")
            f.write(f"Precision:     {metrics_test['Precision']:.6f}\n")
            f.write(f"Recall:        {metrics_test['Recall']:.6f}\n")
            f.write(f"F1-Score:      {metrics_test['F1_Score']:.6f}\n")
            f.write(f"Specificity:   {metrics_test['Specificity']:.6f}\n\n")
            
            f.write(f"Matriz de Confusão (Teste):\n")
            f.write(f"  True Negatives:  {metrics_test['True_Negatives']:,}\n")
            f.write(f"  False Positives: {metrics_test['False_Positives']:,}\n")
            f.write(f"  False Negatives: {metrics_test['False_Negatives']:,}\n")
            f.write(f"  True Positives:  {metrics_test['True_Positives']:,}\n\n")
            
            # Comparação Train vs Test
            f.write("COMPARAÇÃO TRAIN vs TEST (OVERFITTING CHECK)\n")
            f.write("-" * 90 + "\n")
            delta_acc = metrics_train['Accuracy'] - metrics_test['Accuracy']
            delta_auc = metrics_train['AUC_ROC'] - metrics_test['AUC_ROC']
            delta_f1 = metrics_train['F1_Score'] - metrics_test['F1_Score']
            
            status_acc = "✓ OK" if delta_acc <= 0.05 else " OVERFITTING LEVE" if delta_acc <= 0.10 else "❌ OVERFITTING SEVERO"
            status_auc = "✓ OK" if delta_auc <= 0.05 else " OVERFITTING LEVE" if delta_auc <= 0.10 else "❌ OVERFITTING SEVERO"
            status_f1 = "✓ OK" if delta_f1 <= 0.05 else "OVERFITTING LEVE" if delta_f1 <= 0.10 else "❌ OVERFITTING SEVERO"
            
            f.write(f"Δ Accuracy:  {delta_acc:+.6f}  {status_acc}\n")
            f.write(f"Δ AUC-ROC:   {delta_auc:+.6f}  {status_auc}\n")
            f.write(f"Δ F1-Score:  {delta_f1:+.6f}  {status_f1}\n\n")
            
            # Taxa de detecção
            f.write("TAXA DE DETECÇÃO DE CHURN\n")
            f.write("-" * 90 + "\n")
            det_rate_train = metrics_train['True_Positives'] / (metrics_train['True_Positives'] + metrics_train['False_Negatives']) if (metrics_train['True_Positives'] + metrics_train['False_Negatives']) > 0 else 0
            det_rate_test = metrics_test['True_Positives'] / (metrics_test['True_Positives'] + metrics_test['False_Negatives']) if (metrics_test['True_Positives'] + metrics_test['False_Negatives']) > 0 else 0
            
            f.write(f"Treino: {det_rate_train*100:.1f}% ({metrics_train['True_Positives']:,} de {metrics_train['True_Positives'] + metrics_train['False_Negatives']:,})\n")
            f.write(f"Teste:  {det_rate_test*100:.1f}% ({metrics_test['True_Positives']:,} de {metrics_test['True_Positives'] + metrics_test['False_Negatives']:,})\n\n")
            
        print(f"   ✓ Salvo em: {report_path}")
    
    def save_anova_report(self, anova_df: pd.DataFrame, filename: str = "anova_features.txt"):
        """Salva ANOVA em TXT formatado"""
        print(f" Salvando ANOVA...")
        
        report_path = self.output_dir / filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("ANOVA F-TEST - ANÁLISE DE SIGNIFICÂNCIA DAS FEATURES\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"{'RANK':<6} | {'FEATURE':<40} | {'F-STATISTIC':>15} | {'P-VALUE':>15} | {'ETA²':>10} | {'SIG':>5}\n")
            f.write("-" * 100 + "\n")
            
            for _, row in anova_df.iterrows():
                f.write(f"{row['Rank']:<6} | {row['Feature']:<40} | {row['F_Statistic']:>15.4f} | {row['P_Value']:>15.2e} | {row['Eta_Squared']:>10.6f} | {row['Significance']:>5}\n")
            
            f.write("\n" + "=" * 100 + "\n")
            f.write("LEGENDA\n")
            f.write("=" * 100 + "\n")
            f.write("F-Statistic: Valor do teste F (maior = mais significante)\n")
            f.write("P-Value:     Valor-p (< 0.05 = significante)\n")
            f.write("Eta²:        Tamanho do efeito (0-1, quanto da variação é explicada)\n")
            f.write("SIG:         *** p<0.001 | ** p<0.01 | * p<0.05 | NS = Não significante\n")
        
        print(f"   ✓ Salvo em: {report_path}")
    
    def save_correlations_report(self, corr_df: pd.DataFrame, filename: str = "correlacoes_features.txt"):
        """Salva correlações em TXT formatado"""
        print(f"Salvando correlações...")
        
        report_path = self.output_dir / filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("CORRELAÇÃO DE PEARSON - FEATURES vs TARGET\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"{'RANK':<6} | {'FEATURE':<40} | {'CORRELAÇÃO':>15} | {'ABS(CORR)':>15}\n")
            f.write("-" * 100 + "\n")
            
            for _, row in corr_df.iterrows():
                corr_indicator = "🔴" if row['Correlation'] > 0.3 else ("🟢" if row['Correlation'] < -0.3 else "⚪")
                f.write(f"{row['Rank']:<6} | {row['Feature']:<40} | {row['Correlation']:>15.6f} | {row['Abs_Correlation']:>15.6f} {corr_indicator}\n")
            
            f.write("\n" + "=" * 100 + "\n")
            f.write("INTERPRETAÇÃO\n")
            f.write("=" * 100 + "\n")
            f.write(" Correlação > 0.3:  Correlação positiva forte com churn\n")
            f.write(" Correlação < -0.3: Correlação negativa forte com churn\n")
            f.write("⚪ |Correlação| < 0.3: Correlação fraca\n")
        
        print(f"   ✓ Salvo em: {report_path}")
    
    def save_feature_importance(self, top_k: int = 20, filename: str = "feature_importance.csv"):
        """Salva feature importance do modelo"""
        print(f" Salvando feature importance...")
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        importance_df = pd.DataFrame({
            'Rank': range(1, len(importances) + 1),
            'Feature': [self.feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
        
        importance_path = self.output_dir / filename
        importance_df.to_csv(importance_path, index=False)
        print(f"   ✓ Salvo em: {importance_path}")
        
        return importance_df[:top_k]
    
    def save_all_reports(self, anova_df: pd.DataFrame, corr_df: pd.DataFrame, metrics_train: Dict, metrics_test: Dict):
        """Salva todos os relatórios"""
        self.save_metrics_report(metrics_train, metrics_test)
        self.save_anova_report(anova_df)
        self.save_correlations_report(corr_df)
        self.save_feature_importance()
    
    def train(self) -> Dict:
        """Pipeline completo de treinamento"""
        print("=" * 90)
        print(" TREINAMENTO GRADIENT BOOSTING CHURN")
        print("=" * 90 + "\n")
        
        # 1. Carregar dados
        print("[1/8] Carregando dados...")
        df_train, df_test = self.load_data()
        
        # 2. Preparar features
        print("\n[2/8]  Preparando features...")
        self.prepare_features(df_train, df_test)
        
        # 3. Treinar modelo
        print("\n[3/8]  Treinando modelo...")
        self.fit_model()
        
        # 4. Avaliar
        print("\n[4/8]  Avaliando modelo...")
        metrics_train = self.evaluate(self.X_train, self.y_train, "Treino")
        metrics_test = self.evaluate(self.X_test, self.y_test, "Teste")
        
        # 5. ANOVA
        print("\n[5/8] Calculando ANOVA...")
        anova_df = self.calculate_anova()
        
        # 6. Correlações
        print("\n[6/8] Calculando correlações...")
        corr_df = self.calculate_correlations()
        
        # 7. Salvar relatórios
        print("\n[7/8]  Salvando relatórios...")
        self.save_all_reports(anova_df, corr_df, metrics_train, metrics_test)
        
        # 8. Salvar modelo
        print("\n[8/8]  Salvando modelo...")
        model_path = self.output_dir / "gradient_boosting_model.joblib"
        joblib.dump(self.model, model_path)
        print(f"   ✓ Modelo salvo em: {model_path}")
        
        print("\n" + "=" * 90)
        print(" TREINAMENTO CONCLUÍDO!")
        print("=" * 90)
        
        return {
            "metrics_train": metrics_train,
            "metrics_test": metrics_test,
            "anova": anova_df,
            "correlations": corr_df,
            "model": self.model
        }


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == '__main__':
    trainer = GradientBoostingTrainerCustom(
        train_path='data/processed/train.xlsx',
        test_path='data/processed/test.xlsx',
        n_estimators=200,          # Mais árvores com menos poder cada uma
        learning_rate=0.05,        # Taxa menor = aprendizado mais lento e controlado
        max_depth=3,               # Árvores rasas = menos memorização
        subsample=0.8,             # Usa 80% das amostras = regularização
        min_samples_split=20,      # Divide só se tiver 20+ amostras
        min_samples_leaf=10,       # Folhas precisam de 10+ amostras
        output_dir='outputs'
    )
    
    results = trainer.train()