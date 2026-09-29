from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

def train_model(df: pd.DataFrame, train_until_year: int = 2022, test_year: int = 2023):
    """
    Entrena dos modelos independientes de Gradient Boosting:
    1. model_win: Predice si el piloto ganará la carrera (positionOrder == 1).
    2. model_podium: Predice si el piloto subirá al podio (positionOrder <= 3).
    """
    # Definir etiquetas de entrenamiento basadas en los resultados reales
    df['win'] = (df['positionOrder'] == 1).astype(int)
    df['podium'] = (df['positionOrder'] <= 3).astype(int)
    
    # Variables predictoras (features) de ingeniería de datos
    features = [
        'grid', 
        'position_qualifying', 
        'driver_recent_avg_pos', 
        'constructor_recent_avg_points', 
        'grid_vs_recent_form'
    ]
    
    # Filtrar datos de entrenamiento históricos hasta el año indicado
    train_data = df[df['year'] <= train_until_year].dropna(subset=features + ['win', 'podium'])
    
    X_train = train_data[features]
    y_win = train_data['win']
    y_podium = train_data['podium']
    
    # Instanciar y entrenar el modelo de Victoria (P1)
    model_win = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, 
        random_state=42
    )
    model_win.fit(X_train, y_win)
    
    # Instanciar y entrenar el modelo de Podio (Top 3)
    model_podium = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, 
        random_state=42
    )
    model_podium.fit(X_train, y_podium)
    
    return model_win, model_podium

