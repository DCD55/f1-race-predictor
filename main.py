from src.data_loader import load_and_merge_data
from src.features import F1FeaturePipeline
from src.model import train_model

def main():
    print("1. Cargando y cruzando archivos de Kaggle...")
    raw_df = load_and_merge_data()

    print("2. Calculando métricas de rendimiento reciente (Feature Engineering)...")
    pipeline = F1FeaturePipeline(raw_df)
    processed_df = pipeline.process()

    print("3. Entrenando el modelo de Gradient Boosting...")
    model = train_model(processed_df, train_until_year=2022, test_year=2023)
    
    print("¡Pipeline ejecutado con éxito!")

if __name__ == "__main__":
    main()
