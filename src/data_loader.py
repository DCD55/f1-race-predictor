import pandas as pd

def load_and_merge_data(data_path: str = "data/") -> pd.DataFrame:
    races = pd.read_csv(f"{data_path}races.csv")
    results = pd.read_csv(f"{data_path}results.csv")
    qualifying = pd.read_csv(f"{data_path}qualifying.csv")
    drivers = pd.read_csv(f"{data_path}drivers.csv")
    constructors = pd.read_csv(f"{data_path}constructors.csv")

    # AÑADIDO: Incluimos 'name' en la lista de columnas de races
    df = results.merge(races[['raceId', 'year', 'round', 'circuitId', 'name', 'date']], on='raceId', how='left')
    df = df.merge(drivers[['driverId', 'driverRef', 'dob']], on='driverId', how='left')
    df = df.merge(constructors[['constructorId', 'constructorRef']], on='constructorId', how='left')

    qualifying_clean = qualifying[['raceId', 'driverId', 'position']].rename(columns={'position': 'position_qualifying'})
    df = df.merge(qualifying_clean, on=['raceId', 'driverId'], how='left')

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['date', 'round', 'positionOrder']).reset_index(drop=True)
    
    return df
