import pandas as pd

import pandas as pd

class F1FeaturePipeline:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def process(self) -> pd.DataFrame:
        # Variable objetivo: Podio (Top 3)
        self.df['target_podium'] = (self.df['positionOrder'] <= 3).astype(int)

        # Forma reciente del piloto: Promedio móvil de posición (últimas 5 carreras)
        self.df['driver_recent_avg_pos'] = self.df.groupby('driverId')['positionOrder'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )

        # Rendimiento del equipo: Promedio móvil de puntos (últimas 3 carreras)
        self.df['constructor_recent_avg_points'] = self.df.groupby('constructorId')['points'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )

        # Delta entre parrilla de salida y forma reciente
        self.df['grid_vs_recent_form'] = self.df['grid'] - self.df['driver_recent_avg_pos']

        # Rellenar valores nulos iniciales
        self.df.fillna({
            'driver_recent_avg_pos': 15,
            'constructor_recent_avg_points': 0,
            'grid_vs_recent_form': 0,
            'position_qualifying': 20
        }, inplace=True)

        return self.df
