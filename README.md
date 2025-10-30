# 🔍 Sistema de Detecção de Fraudes em Cartão de Crédito

Sistema completo de Machine Learning para detectar fraudes em transações de cartão de crédito, implementado desde a análise exploratória até o modelo em produção.
Lembrando que o sistema foi feito para aprendizado de Machine Learning, análise de dados entre outras tecnologias, ou seja, o código inteiro foi feito inteiramente à mão sem "copia e cola".

## 🎯 Objetivo

Desenvolver um sistema robusto de detecção de fraudes com foco em:
- **Recall Alto (>85%)**: Capturar máximo de fraudes possível
- **Precisão Balanceada**: Minimizar falsos positivos
- **Produção Ready**: Código modular e reutilizável
- **Performance**: Predições em tempo real

## 📊 Dataset

- **Fonte**: Transações de cartão de crédito europeias
- **Tamanho**: 284.807 transações
- **Desbalanceamento**: 99.83% legítimas vs 0.17% fraudes (492 fraudes)
- **Features**: 
  - 28 features anônimas (V1-V28) transformadas por PCA
  - `Time`: segundos desde primeira transação
  - `Amount`: valor da transação
  - `Class`: 0 (legítima) ou 1 (fraude)

## 🛠️ Stack Tecnológica

### Core
- **Python 3.14+**
- **Pandas & NumPy**: Manipulação e análise de dados
- **Scikit-learn**: Pré-processamento e métricas
- **XGBoost**: Modelo de gradient boosting

### Balanceamento & Processamento
- **SMOTE (imblearn)**: Oversampling sintético
- **StandardScaler**: Normalização de features
- **LabelEncoder**: Encoding de categorias

### Visualização & Análise
- **Matplotlib & Seaborn**: Gráficos e análises visuais
- **Jupyter Notebook**: Exploração interativa

### Persistência
- **Pickle**: Serialização de modelos

## 🔄 Pipeline de ML

### 1. **Análise Exploratória (EDA)**
```python
notebooks/exploracao_dados.ipynb
```
- Distribuição temporal de fraudes (0-6h, 6-12h, etc.)
- Correlação de features com fraudes (V14, V17 mais relevantes)
- Análise de valores de transações (fraudes tendem a valores baixos)
- Visualizações de padrões e outliers

### 2. **Feature Engineering**
```python
src/fraud_detector.py → create_features()
```

**Features Numéricas:**
- `Amount_log`: `np.log1p(Amount)` - reduz impacto de outliers
- `Amount_sqrt`: `√Amount` - suaviza distribuição
- `Time_hours`: conversão de segundos para horas

**Features Temporais Cíclicas:**
- `Time_sin`: `sin(2π × Time_hours / 24)` - padrão circular diário
- `Time_cos`: `cos(2π × Time_hours / 24)` - complemento seno

**Features de Interação:**
- `V14_V17`: multiplicação de features importantes
- `Amount_V14`: interação valor × V14

**Features Categóricas:**
- `Amount_category`: [Muito baixo, Baixo, Médio, Alto, Muito Alto]

### 3. **Pré-processamento**
```python
src/fraud_detector.py → preprocess()
```

**Sequência:**
1. **Feature Engineering** (create_features)
2. **Label Encoding** (Amount_category → números)
3. **Remoção** de features não numéricas
4. **Divisão** treino/teste (80/20 com stratify)
5. **SMOTE** (227k legítimas → 227k fraudes sintéticas)
6. **StandardScaler** (média=0, desvio=1)

### 4. **Modelagem**

**Arquitetura XGBoost:**
```python
XGBClassifier(
    n_estimators=50,      # 50 árvores sequenciais
    max_depth=6,          # Profundidade máxima
    learning_rate=0.1,    # Taxa de aprendizado (10%)
    random_state=42,      # Reprodutibilidade
    eval_metric='logloss' # Métrica de avaliação
)
```

**Por que XGBoost?**
- Ensemble de árvores que corrigem erros anteriores
- Regularização embutida (evita overfitting)
- Excelente para dados desbalanceados
- Rápido e eficiente

## 📈 Resultados

### Métricas Finais
| Métrica | Valor |
|---------|-------|
| **Recall** | ~85-90% |
| **Precision** | ~28-35% |
| **F1-Score** | ~42-48% |

### Interpretação
- ✅ **Recall Alto**: Detecta maioria das fraudes (prioridade)
- ⚠️ **Precision Baixa**: Alguns falsos positivos (aceitável para fraudes)
- 💡 **Trade-off**: Melhor bloquear fraude legítima que deixar passar fraude real

### Matriz de Confusão (Exemplo)
```
                Pred Legítima  Pred Fraude
Real Legítima      56,800         62
Real Fraude           15          83
```
- 83/98 fraudes detectadas (84.7% recall)
- 62 clientes honestos bloqueados (falsos positivos)

## 📁 Estrutura do Projeto

```
fraudcard/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv          # Dataset original
│   └── processed/                  # (vazio - para dados processados)
│
├── notebooks/
│   └── exploracao_dados.ipynb      # EDA completa + análise
│
├── src/
│   ├── __init__.py                 # Torna src um módulo
│   ├── fraud_detector.py           # Classe principal do modelo
│   └── sample_transaction.py       # Transação de exemplo
│
├── models/
│   └── fraud_detector.pkl          # Modelo treinado salvo
│
├── main.py                         # Script de execução
├── requirements.txt                # Dependências
└── README.md                       # Este arquivo
```

## 🚀 Como Executar

### 1. **Instalação**

```bash
# Clone o repositório
git clone https://github.com/VitorSaviolli/Fraud-Detection.git
cd Fraud-Detection

# Instale as dependências
pip install -r requirements.txt
```

### 2. **Treinamento**

```bash
# Treina modelo, salva e testa predição
python main.py
```

**Output esperado:**
```
Sistema de Detecção de Fraudes:
Carregando dados...
Criando features...
Separando X e y...
Dividindo treino e teste...
Aplicando SMOTE...
Normalizando...
Treinando XGBOOST...
Avaliando...
Recall 0.857 (85.7%)
Precision 0.286 (28.6%)
Modelo salvo em models/fraud_detector.pkl

Testando predição...
Resultado: LEGÍTIMA
Probabilidade de Fraude: 0.084
Probabilidade Legítima: 0.916
```

### 3. **Usando o Modelo**

```python
from src.fraud_detector import FraudDetector

# Carrega modelo treinado
detector = FraudDetector()
detector.load_model('models/fraud_detector.pkl')

# Nova transação
transacao = {
    'Time': 406,
    'V1': -1.359807,
    'V2': -0.072781,
    # ... V3-V28
    'Amount': 149.62
}

# Predição
resultado = detector.predict(transacao)
print(resultado)
# {'prediction': 'LEGÍTIMA', 
#  'probability_fraud': 0.084, 
#  'probability_legit': 0.916}
```

## 🔍 Detalhes Técnicos

### Classe FraudDetector

```python
class FraudDetector:
    def __init__(self)
    def create_features(df)      # Feature engineering
    def preprocess(df, fit)       # Pré-processamento
    def train(csv_path)           # Treina modelo
    def predict(transaction)      # Prediz fraude
    def save_model(filepath)      # Salva modelo
    def load_model(filepath)      # Carrega modelo
```

### Fluxo de Predição

```
Nova Transação (dict)
    ↓
create_features()  → Engenharia de features
    ↓
preprocess()       → Label encoding + seleção
    ↓
scaler.transform() → Normalização
    ↓
model.predict()    → Classe (0 ou 1)
    ↓
model.predict_proba() → Probabilidades
    ↓
return {prediction, probability_fraud, probability_legit}
```

## 📚 Conceitos Aplicados

### SMOTE (Synthetic Minority Over-sampling)
- Cria fraudes sintéticas interpolando entre vizinhos
- Balanceia dataset (50/50) sem overfitting
- Melhora detecção de classe minoritária

### StandardScaler
- Normaliza features (média=0, desvio=1)
- Evita dominância de features com valores altos
- Essencial para algoritmos baseados em distância

### Stratified Split
- Mantém proporção de classes em treino/teste
- Garante representatividade estatística
- Essencial para dados desbalanceados

### Feature Engineering Temporal
- `sin/cos` capturam padrões cíclicos (horários)
- Melhor que features lineares para tempo
- Modelo entende que 23h está perto de 0h

## 🎓 Aprendizados

**Técnicos:**
- Tratamento de dados extremamente desbalanceados
- Feature engineering criativa com PCA
- Trade-offs entre Recall e Precision
- Serialização e deploy de modelos ML

**Conceituais:**
- Importância de Recall em detecção de fraude
- Custo de falsos positivos vs falsos negativos
- Interpretabilidade vs Performance
- Pipeline completo de ML (EDA → Produção)


## 📄 Licença

Este projeto é open-source para fins educacionais.

---

⭐ **Se este projeto te ajudou ou achou interessante, deixe uma estrela no repositório, me ajuda bastante!**