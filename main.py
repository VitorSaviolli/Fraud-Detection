from src.sample_transaction import sample_transaction
from src.fraud_detector import FraudDetector
def main():
    print('Sistema de Deteção de Fraudes:')


    detector = FraudDetector()
    
    recall, precision = detector.train('data/raw/creditcard.csv')

    detector.save_model('models/fraud_detector.pkl')

    print("\n Testando predição...")

    result = detector.predict(sample_transaction)
    print(f"Resultados: {result['prediction']}")
    print(f"Probabilidade de Fraude: {result['probability_fraud']:.3f}")
    print(f"Probabilidade Legítima: {result['probability_legit']:.3f}")

if __name__ == "__main__":
    main()