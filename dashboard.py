import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import load_and_merge_data
from src.features import F1FeaturePipeline
from src.model import train_model

st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️", layout="wide")

NATIONALITY_FLAGS = {
    'British': '🇬🇧', 'Dutch': '🇳🇱', 'Mexican': '🇲🇽', 'Monegasque': '🇲🇨', 
    'Spanish': '🇪🇸', 'Australian': '🇦🇺', 'French': '🇫🇷', 'Thai': '🇹🇭', 
    'Japanese': '🇯🇵', 'Finnish': '🇫🇮', 'Chinese': '🇨🇳', 'German': '🇩🇪', 
    'Danish': '🇩🇰', 'Canadian': '🇨🇦', 'American': '🇺🇸', 'Brazilian': '🇧🇷', 
    'Italian': '🇮🇹', 'Polish': '🇵🇱', 'Argentine': '🇦🇷', 'Colombian': '🇨🇴'
}

CARS = {
    'red_bull': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2023/red-bull-racing.png',
    'ferrari': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2023/ferrari.png',
    'mercedes': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2023/mercedes.png',
    'mclaren': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2023/mclaren.png',
    'aston_martin': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2023/aston-martin.png',
    'alpine': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2023/alpine.png',
    'williams': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2023/williams.png',
    'alpha_tauri': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2023/alphatauri.png',
    'alphatauri': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2023/alphatauri.png',
    'rb': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/rb.png',
    'sauber': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/kick-sauber.png',
    'alfa': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2023/alfa-romeo.png',
    'haas': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2023/haas-f1-team.png',
    'renault': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2020/renault.png',
    'racing_point': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2020/racing-point.png',
    'toro_rosso': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2019/toro-rosso.png',
    'force_india': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2018/force-india.png',
    'mclaren_renault': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2020/mclaren.png',
    'mclaren_mercedes': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2023/mclaren.png',
    'ferrari_haas': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2023/haas-f1-team.png',
    'lotus_f1': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2015/lotus.png',
    'manor': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2016/manor.png',
    'sauber_f1': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2018/sauber.png',
    'default': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2023/williams.png'
}

COLORS = {
    'red_bull': '#3671C6', 'ferrari': '#E80020', 'mercedes': '#27F4D2',
    'mclaren': '#FF8000', 'aston_martin': '#229971', 'alpine': '#FF87BC',
    'williams': '#64C4FF', 'alpha_tauri': '#5E8FAA', 'alphatauri': '#5E8FAA',
    'rb': '#6692FF', 'sauber': '#52E252', 'alfa': '#C92D4B',
    'haas': '#B6BABD', 'renault': '#FFF500', 'racing_point': '#F596C8',
    'toro_rosso': '#469BFF', 'force_india': '#F596C8', 'lotus_f1': '#C5A059',
    'manor': '#313538', 'sauber_f1': '#9B0000', 'mclaren_renault': '#FF8000',
    'mclaren_mercedes': '#FF8000', 'ferrari_haas': '#B6BABD', 'default': '#FFFFFF'
}

TEAM_LOGOS = {
    'red_bull': 'https://media.formula1.com/content/dam/fom-website/teams/2024/red-bull-racing-logo.png',
    'ferrari': 'https://media.formula1.com/content/dam/fom-website/teams/2024/ferrari-logo.png',
    'mercedes': 'https://media.formula1.com/content/dam/fom-website/teams/2024/mercedes-logo.png',
    'mclaren': 'https://media.formula1.com/content/dam/fom-website/teams/2024/mclaren-logo.png',
    'aston_martin': 'https://media.formula1.com/content/dam/fom-website/teams/2024/aston-martin-logo.png',
    'alpine': 'https://media.formula1.com/content/dam/fom-website/teams/2024/alpine-logo.png',
    'williams': 'https://media.formula1.com/content/dam/fom-website/teams/2024/williams-logo.png',
    'alpha_tauri': 'https://media.formula1.com/content/dam/fom-website/teams/2023/alphatauri-logo.png',
    'alphatauri': 'https://media.formula1.com/content/dam/fom-website/teams/2023/alphatauri-logo.png',
    'rb': 'https://media.formula1.com/content/dam/fom-website/teams/2024/rb-logo.png',
    'sauber': 'https://media.formula1.com/content/dam/fom-website/teams/2024/kick-sauber-logo.png',
    'alfa': 'https://media.formula1.com/content/dam/fom-website/teams/2023/alfa-romeo-logo.png',
    'haas': 'https://media.formula1.com/content/dam/fom-website/teams/2024/haas-f1-team-logo.png',
    'renault': 'https://media.formula1.com/content/dam/fom-website/teams/2020/renault-logo.png',
    'racing_point': 'https://media.formula1.com/content/dam/fom-website/teams/2020/racing-point-logo.png',
    'toro_rosso': 'https://media.formula1.com/content/dam/fom-website/teams/2019/toro-rosso-logo.png',
    'force_india': 'https://media.formula1.com/content/dam/fom-website/teams/2018/force-india-logo.png',
    'lotus_f1': 'https://media.formula1.com/content/dam/fom-website/teams/2015/lotus-logo.png',
    'manor': 'https://media.formula1.com/content/dam/fom-website/teams/2016/manor-logo.png',
    'sauber_f1': 'https://media.formula1.com/content/dam/fom-website/teams/2018/sauber-logo.png',
    'mclaren_renault': 'https://media.formula1.com/content/dam/fom-website/teams/2020/mclaren-logo.png',
    'mclaren_mercedes': 'https://media.formula1.com/content/dam/fom-website/teams/2024/mclaren-logo.png',
    'ferrari_haas': 'https://media.formula1.com/content/dam/fom-website/teams/2024/haas-f1-team-logo.png',
    'default': 'https://media.formula1.com/content/dam/fom-website/teams/2024/williams-logo.png'
}

CIRCUIT_MAPS = {
    1: 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Bahrain_Circuit.png',
    2: 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Saudi_Arabia_Circuit.png',
    3: 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Australia_Circuit.png',
    4: 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Azerbaijan_Circuit.png',
    5: 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Miami_Circuit.png',
    6: 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Monaco_Circuit.png',
    7: 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Spain_Circuit.png',
    8: 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Canada_Circuit.png',
    9: 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Austria_Circuit.png',
    10: 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Great_Britain_Circuit.png',
    11: 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Hungary_Circuit.png',
    12: 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Belgium_Circuit.png',
    13: 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Netherlands_Circuit.png',
    14: 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Italy_Circuit.png',
    15: 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Singapore_Circuit.png',
    16: 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Japan_Circuit.png',
    17: 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Qatar_Circuit.png',
    18: 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/USA_Circuit.png',
    19: 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Mexico_Circuit.png',
    20: 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Brazil_Circuit.png',
    21: 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Las_Vegas_Circuit.png',
    22: 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Abu_Dhabi_Circuit.png',
    'default': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Bahrain_Circuit.png'
}

@st.cache_data
def get_driver_details():
    drivers_df = pd.read_csv("data/drivers.csv")
    nats = drivers_df.set_index('driverRef')['nationality'].to_dict()
    nums = {}
    for _, row in drivers_df.iterrows():
        num = row['number']
        if pd.notna(num) and str(num).strip() != '\\N':
            nums[row['driverRef']] = str(int(float(num)))
        else:
            nums[row['driverRef']] = ""
    return nats, nums

driver_nats, driver_nums = get_driver_details()

st.markdown("""
    <style>
        .stApp { background-color: #1e1e24; color: #ffffff; }
        
        .block-container {
            max-width: 100% !important;
            padding-left: 3rem !important;
            padding-right: 2rem !important;
        }

        .driver-card {
            background-color: transparent; 
            padding: 0px; 
            border-radius: 0px;
            box-shadow: none;
            text-align: center; 
            height: 100%;
        }
        
        .driver-badge-box {
            background-color: #1a1a1e;
            border-radius: 6px;
            padding: 10px 12px;
            margin-bottom: 12px;
            border-top: 4px solid #E10600;
            box-shadow: 0 4px 10px rgba(0,0,0,0.5);
            height: 68px;
            display: flex;
            align-items: center;
        }

        .driver-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 6px;
            width: 100%;
            text-align: left;
        }
        .driver-info {
            display: flex;
            flex-direction: column;
            flex-grow: 1;
            overflow: hidden;
        }
        .driver-name { 
            font-size: 0.90rem; 
            font-weight: bold; 
            margin: 0; 
            line-height: 1.2;
            color: #ffffff;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .team-name { 
            font-size: 0.70rem; 
            color: #a0a0a0; 
            margin: 0; 
            text-transform: uppercase; 
            line-height: 1.2;
        }
        .driver-number {
            font-size: 1.6rem;
            font-weight: 900;
            color: #ffffff;
            font-family: 'Impact', 'Arial Black', sans-serif;
            font-style: italic;
            letter-spacing: -1px;
            line-height: 1;
            margin-right: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.6);
            min-width: 32px;
            text-align: center;
        }

        .uniform-label {
            font-size: 0.82rem; color: #a0a0a0; text-align: center; font-weight: bold; 
            letter-spacing: 0.5px; margin-top: 10px; margin-bottom: 4px;
        }
        .uniform-value {
            font-size: 0.82rem; color: #ffffff; text-align: center; font-weight: bold; 
            margin-top: 2px; margin-bottom: 6px;
        }
        .stat-box {
            background-color: #1a1a1e; padding: 7px; border-radius: 4px;
            margin-top: 6px; font-size: 0.82rem; color: #cccccc; text-align: left; font-weight: 500;
        }
        hr { border-top: 1px solid #444; margin: 8px 0; }
        
        .f1-footer {
            text-align: center;
            color: #55555d;
            font-size: 0.75rem;
            margin-top: 4rem;
            margin-bottom: 1rem;
            letter-spacing: 1px;
        }
        .f1-footer a {
            color: #666675;
            text-decoration: none;
            transition: color 0.2s ease;
        }
        .f1-footer a:hover {
            color: #E10600;
        }
    </style>
""", unsafe_allow_html=True)

def create_donut_chart(prob, color):
    val = round(prob * 100, 1)
    df_pie = pd.DataFrame({
        'Category': ['Active', 'Rest'],
        'Value': [val, max(0.0, 100.0 - val)]
    })
    fig = px.pie(
        df_pie, values='Value', names='Category', hole=0.65,
        color='Category', color_discrete_map={'Active': color, 'Rest': '#1a1a1e'}
    )
    fig.update_traces(textinfo='none', hoverinfo='skip')
    fig.update_layout(
        showlegend=False, height=135,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        annotations=[{
            'text': f"{val}%", 
            'font': {'size': 16, 'color': 'white', 'family': 'Arial', 'weight': 'bold'},
            'showarrow': False, 'x': 0.5, 'y': 0.5
        }]
    )
    return fig

@st.cache_resource
def load_and_train():
    raw_df = load_and_merge_data()
    pipeline = F1FeaturePipeline(raw_df)
    df = pipeline.process()
    model_w, model_p = train_model(df, train_until_year=2022, test_year=2023)
    return df, model_w, model_p

with st.spinner('Processing historical data and training models...'):
    df, model_win, model_podium = load_and_train()

col_inputs, col_spacer, col_map = st.columns([1.1, 0.35, 1.15])

with col_inputs:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=160)
    st.title("Grid Prediction - Top 5")
    
    sub_col1, sub_col2 = st.columns(2)
    available_years = sorted(df['year'].dropna().unique(), reverse=True)
    with sub_col1:
        selected_year = st.selectbox("Season:", available_years, index=0)
    
    races_in_year = df[df['year'] == selected_year][['round', 'name', 'circuitId']].drop_duplicates().sort_values('round')
    with sub_col2:
        selected_gp_name = st.selectbox("Grand Prix:", races_in_year['name'].tolist())

selected_round = races_in_year[races_in_year['name'] == selected_gp_name]['round'].values[0]
current_circuit_id = races_in_year[races_in_year['name'] == selected_gp_name]['circuitId'].values[0]

with col_spacer:
    pass

with col_map:
    circuit_img_url = CIRCUIT_MAPS.get(current_circuit_id, CIRCUIT_MAPS['default'])
    st.markdown('<div style="display: flex; justify-content: flex-end; margin-top: 5px;">', unsafe_allow_html=True)
    st.image(circuit_img_url, width=460)
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

race_data = df[(df['year'] == selected_year) & (df['round'] == selected_round)].copy()

if not race_data.empty:
    features = ['grid', 'position_qualifying', 'driver_recent_avg_pos', 'constructor_recent_avg_points', 'grid_vs_recent_form']
    
    race_data['win_prob'] = model_win.predict_proba(race_data[features])[:, 1]
    race_data['podium_prob'] = model_podium.predict_proba(race_data[features])[:, 1]
    
    top5 = race_data.sort_values(by='win_prob', ascending=False).head(5).reset_index(drop=True)
    cols = st.columns(5)
    
    for col, (_, row) in zip(cols, top5.iterrows()):
        driver_ref = row['driverRef']
        team_ref = row['constructorRef']
        p_win = row['win_prob']
        p_podium = row['podium_prob']
        
        nat = driver_nats.get(driver_ref, 'Unknown')
        flag = NATIONALITY_FLAGS.get(nat, '')
        
        d_num = driver_nums.get(driver_ref, "")
        num_display = f"{d_num}" if d_num else ""
        
        car_img = CARS.get(team_ref, CARS.get(str(team_ref).lower(), CARS['default']))
        team_logo = TEAM_LOGOS.get(team_ref, TEAM_LOGOS.get(str(team_ref).lower(), TEAM_LOGOS['default']))
        team_color = COLORS.get(team_ref, COLORS.get(str(team_ref).lower(), COLORS['default']))
        
        season_history = df[(df['year'] == selected_year) & (df['round'] < selected_round) & (df['driverRef'] == driver_ref)]
        season_points = season_history['points'].sum() if not season_history.empty else 0
        
        circuit_history = df[(df['circuitId'] == current_circuit_id) & (df['year'] < selected_year) & (df['driverRef'] == driver_ref)]
        avg_circuit_pos = circuit_history['positionOrder'].mean() if not circuit_history.empty else None
        circuit_str = f"P{avg_circuit_pos:.1f} Avg." if avg_circuit_pos else "No prior record"
        
        podium_val_pct = float(p_podium * 100)
        
        with col:
            st.markdown(f"""
                <div class="driver-card">
                    <div class="driver-badge-box" style="border-top-color: {team_color};">
                        <div class="driver-header">
                            <div class="driver-number">{num_display}</div>
                            <div class="driver-info">
                                <p class="driver-name">{flag} {str(driver_ref).upper()}</p>
                                <p class="team-name">{str(team_ref).replace("_", " ")}</p>
                            </div>
                            <img src="{team_logo}" style="height: 32px; width: auto; object-fit: contain; margin-left: auto;">
                        </div>
                    </div>
            """, unsafe_allow_html=True)
            
            # Imagen del coche de F1
            st.image(car_img, use_container_width=True)
            
            st.markdown('<hr>', unsafe_allow_html=True)
            
            st.markdown('<p class="uniform-label">WIN PROBABILITY</p>', unsafe_allow_html=True)
            st.plotly_chart(create_donut_chart(p_win, team_color), use_container_width=True)
            
            st.markdown('<p class="uniform-label">PODIUM PROBABILITY</p>', unsafe_allow_html=True)
            st.markdown(f"""
                <div style="background-color: #1a1a1e; border-radius: 4px; width: 100%; height: 10px; margin-bottom: 4px; overflow: hidden;">
                    <div style="background-color: {team_color}; width: {podium_val_pct}%; height: 100%; border-radius: 4px;"></div>
                </div>
                <p class="uniform-value">{podium_val_pct:.1f}%</p>
            """, unsafe_allow_html=True)
            
            st.markdown('<hr>', unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="stat-box"><b>Season Pts:</b> {int(season_points)} pts</div>
                <div class="stat-box"><b>Circuit History:</b> {circuit_str}</div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.warning("No telemetry available for this Grand Prix.")

# Easter egg sutil con "Developed by DCD55" al final del todo
st.markdown("""
    <div class="f1-footer">
        F1 Predictor Engine &bull; Developed by <a href="https://github.com/DCD55" target="_blank">DCD55</a>
    </div>
""", unsafe_allow_html=True)
