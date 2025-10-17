# 🔍 Detecção de Fraudes em Cartão de Crédito(NÃO FINALIZADO!)

Projeto de Machine Learning para detectar fraudes em transações de cartão de crédito com foco em **maximizar o Recall (90%+)** para capturar o máximo de fraudes possível.

## 🎯 Objetivo
- **Recall Alto**: Detectar 90%+ das fraudes (prioridade máxima)
- **Precisão Aceitável**: Minimizar falsos positivos quando possível
- **Impacto**: Proteger clientes e reduzir perdas financeiras

## 📊 Dataset
- **Fonte**: Transações de cartão de crédito
- **Tamanho**: 284.807 transações
- **Classes**: Extremamente desbalanceadas (99.83% legítimas, 0.17% fraudes)
- **Features**: 30 variáveis (V1-V28 + Time + Amount)

## 🛠️ Tecnologias Utilizadas
- **Python**: Linguagem principal
- **Pandas & NumPy**: Manipulação de dados
- **Scikit-learn**: Algoritmos de ML e métricas
- **XGBoost**: Gradient boosting avançado
- **SMOTE**: Balanceamento de classes
- **Matplotlib & Seaborn**: Visualização

## 🔄 Pipeline de ML

### 1. **Exploração de Dados (EDA)**
- Análise de distribuições e padrões temporais
- Identificação de correlações com fraudes
- Visualização de outliers e valores

### 2. **Feature Engineering**
- `Amount_log`: Transformação logarítmica do valor
- `Time_sin/cos`: Features temporais cíclicas
- `V14_V17`: Features de interação
- `Amount_category`: Categorização por faixas

### 3. **Pré-processamento**
- **SMOTE**: Balanceamento sintético (227k exemplos cada classe)
- **StandardScaler**: Normalização das features
- **Train/Test Split**: 80/20 com estratificação

### 4. **Modelagem**
Comparação de 3 algoritmos:
- **Logistic Regression**: Baseline linear
- **Random Forest**: Ensemble de árvores
- **XGBoost**: Gradient boosting otimizado

## 📈 Resultados

| Modelo | Recall | Precision |
|--------|--------|-----------|
| Random Forest | 85.7% | 92.1% |
| XGBoost | 88.2% | 89.5% |
| Logistic Regression | 82.3% | 94.2% |

**🏆 Melhor modelo**: XGBoost (melhor Recall)

## 📁 Estrutura do Projeto
```
capacitron_app/
├── data/
│   └── raw/
│       └── creditcard.csv
├── notebooks/
│   └── exploracao_dados.ipynb
├── src/
├── main.py
└── README.md
```

## 🚀 Como Executar
1. Clone o repositório
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute o notebook: `jupyter notebook notebooks/exploracao_dados.ipynb`

## 📋 Próximos Passos
- [ ] Otimização de threshold para maximizar Recall
- [ ] Hyperparameter tuning com GridSearch
- [ ] Deploy do modelo em produção
- [ ] Monitoramento de performance

## 🎓 Aprendizados
Este projeto foca no **aprendizado prático** de ML, explorando:
- Tratamento de dados desbalanceados
- Feature engineering criativa
- Trade-offs entre Recall e Precision
- Comparação de algoritmos de ML

---
**Autor**: Vitor  Saviolli Gonsalez