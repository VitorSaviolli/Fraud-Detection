import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import recall_score, precision_score
import pickle


class FraudDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.features_names = None

    def create_features(self,df):
        #criador de features iguais ao exploracao_dados
        df = df.copy()

        #features log e raiz quadrada
        df['Amount_log'] = np.log1p(df['Amount'])
        df['Amount_sqrt'] = np.sqrt(df['Amount'])

        #features de tempo
        df['Time_hours'] = df['Time'] / 3600
        df['Time_sin'] = np.sin(2 * np.pi * df['Time_hours'] / 24)

        #features de interação
        #features de feature engineering
        df['V14_V17'] = df['V14'] * df['V17'] #como V14 tem alta chance de ser fraude e V17 tambem, multiplicar os dois aumenta a chance do modelo "pegar" a fraude
        df['Amount_V14'] = df['Amount'] * df['V14']
        

        #categoria de valor
        df['Amount_category'] = pd.cut(
            df['Amount'],
            bins = [0, 10, 50, 100, 500, float('inf')],
            labels=['Muito baixo', 'Baixo', 'Medio', 'Alto', 'Muito Alto']
        )
        return df
    
    def preprocess(self,df,fit=False):
        """Pré-processa os dados (encoding + seleção de features)"""
        #Fit = True, modelo sendo treinado pela primeira vez
        #Fit = False, modelo já treinado sendo utilizado
        df = self.create_features(df)

        if fit:
            self.label_enconder = LabelEncoder()
            df['Amount_category_encoded'] = self.label_enconder.fit_transform(df['Amount_category'])
        else:
            df['Amount_category_encoded'] = self.label_enconder.transform(df['Amount_category'])

        features_to_exclude = ['Time_period', 'Amount_category']
        if 'Class' in df.columns:
            features_to_exclude.append('Class')

        numeric_features = [col for col in df.columns if col not in features_to_exclude]
        df_numeric = df[numeric_features]

        if fit:
            self.feature_names = df_numeric.columns.tolist()
            
        return df_numeric

    def train(self,csv_path):
        """Treina o modelo todo"""
        print("Carregando dados...")
        df = pd.read_csv(csv_path)
    
        print("Criando features...")      #fit = True - > pois é a primeira vez treinando
        df_processed = self.preprocess(df,fit=True)

        print("Separando X e y...")

        X = df_processed
        y = df['Class']

        #stratify -> garante que a proporção de classes sejam as mesmas
        print("Dividindo treino e teste...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y ,test_size=0.2,random_state=42,stratify=y
        )

        print("Aplicando SMOTE...")
        smote = SMOTE(random_state=42)
        #smote cria fraudes sinteticas por exemplo legitimas = 227 e fraudes = 227
        #fit( analisa os dados de fraude para entender padroes)
        #resample -> cria novas fraudes sinteticas ate balancear
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print("Normalizando...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Treinando XGBOOST...")
        self.model = xgb.XGBClassifier(
            n_estimators = 50, #cria 50 arvores , arvore 2 corrige a arvore 1 e assim por diante
            max_depth =6, #profundidade maxima
            learning_rate = 0.1, #aprende 10% por vez, aprende "devagar" mas aprende com cuidado
            random_state = 42, #seed padrao aleatoria
            eval_metric ='logloss' #Metrica de avaliacao 0 -> estudar logloss
        )
        self.model.fit(X_train_scaled, y_train_balanced)

        print("Avaliando...")
        y_pred = self.model.predict(X_test_scaled)
        #dif entre recall e precision
        #Recall -> "Dos casos realmente positivos quantos o modelo encontrou?"
        #Precision -> "Das previsoes positivas quantas estavam certas?"
        recall = recall_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)

        print(f"Recall {recall:.3f} ({recall*100:.1f}%)")
        print(f"Precision {precision:3f} ({precision * 100:.1f}%)")

        return recall,precision

    def predict(self,transaction_dict):
        """Prediz se uma transação é fraude"""
        #Converte o dicionario em DF
        df = pd.DataFrame([transaction_dict])
        #Pré-processa com a função feita -> preprocess()
        df_processed = self.preprocess(df, fit=False)
        #Normaliza
        X_scaled = self.scaler.transform(df_processed)

        #Predição
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]


        return{
            'prediction': 'FRAUDE' if prediction == 1 else 'LEGÍTIMA',
            'probability_fraud': probabilities[1], # exemplo probabilities = [0.92, 0.08] -> [0] pega legitima e [1] pega fraude
            'probability_legit': probabilities[0]

        }

    def save_model(self, filepath):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f: #readbinary , writebinary
            pickle.dump(model_data,f)
        print(f"Modelo salvo em {filepath}")

    def load_model(self,filepath):
        with open(filepath,'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        print(f"Modelo carregado de: {filepath}")



