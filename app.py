import os
import io
import csv
import zipfile
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string, abort
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'omniscient-divine-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///omniscience.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
db = SQLAlchemy(app)

# --- Models (NBA shown as example, add others similarly) ---
class BaseModel:
    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
    def to_prophecy(self):
        d = self.to_dict()
        d['divine_insight'] = self.generate_divine_insight()
        d['future_prediction'] = self.predict_future()
        return d

class NBAStat(db.Model, BaseModel):
    id = db.Column(db.Integer, primary_key=True)
    player_id = db.Column(db.String(50), nullable=False, index=True)
    name = db.Column(db.String(100), nullable=False)
    team = db.Column(db.String(50))
    position = db.Column(db.String(20))
    points = db.Column(db.Float)
    rebounds = db.Column(db.Float)
    assists = db.Column(db.Float)
    steals = db.Column(db.Float)
    blocks = db.Column(db.Float)
    turnovers = db.Column(db.Float)
    minutes_played = db.Column(db.Float)
    season = db.Column(db.String(20), nullable=False, index=True)
    divine_score = db.Column(db.Float)
    prophecy_rating = db.Column(db.Float)
    future_value = db.Column(db.Float)
    def generate_divine_insight(self):
        insights = []
        if self.points and self.points > 25:
            insights.append(f"{self.name} is a scoring deity - {self.points} PPG")
        if self.assists and self.assists > 8:
            insights.append(f"Vision of a playmaking god - {self.assists} APG")
        if self.steals and self.blocks and self.steals + self.blocks > 3:
            insights.append(f"Defensive omnipotence - {self.steals+self.blocks} combined steals/blocks")
        return insights or ["Mortal performance"]
    def predict_future(self):
        if not self.minutes_played: return "Unknown future"
        growth = 1.1 if float(self.season[:4]) < 2023 else 0.98
        return {
            'next_season_points': round((self.points or 0) * growth, 1),
            'peak_season': int(self.season[:4]) + 2,
            'decline_age': 32 if self.position in ['PG', 'SG'] else 34
        }

class Omniscience(db.Model, BaseModel):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, index=True)
    swings_competitive = db.Column(db.Integer)
    percent_swings_competitive = db.Column(db.Float)
    contact = db.Column(db.Integer)
    avg_bat_speed = db.Column(db.Float)
    hard_swing_rate = db.Column(db.Float)
    squared_up_per_bat_contact = db.Column(db.Float)
    squared_up_per_swing = db.Column(db.Float)
    blast_per_bat_contact = db.Column(db.Float)
    blast_per_swing = db.Column(db.Float)
    swing_length = db.Column(db.Float)
    swords = db.Column(db.Integer)
    batter_run_value = db.Column(db.Float)
    whiffs = db.Column(db.Integer)
    whiff_per_swing = db.Column(db.Float)
    batted_ball_events = db.Column(db.Integer)
    batted_ball_event_per_swing = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    cashout_signal = db.Column(db.Boolean, default=False)
    pick_tracked = db.Column(db.Boolean, default=False)
    delta_bat_speed = db.Column(db.Float)
    oscillator_bat_speed = db.Column(db.Float)
    prophecy_rating = db.Column(db.Float)
    future_value = db.Column(db.Float)
    def generate_divine_insight(self):
        if self.oscillator_bat_speed and self.oscillator_bat_speed < -2.5:
            return "DIVINE INTERVENTION: Cashout signal STRONG - bat speed in critical decline"
        if self.blast_per_swing and self.blast_per_swing > 0.35:
            return "Olympian power: Elite blast rate detected"
        if self.whiff_per_swing and self.whiff_per_swing > 0.4:
            return "Mortal weakness: High whiff rate vulnerable to divine pitches"
        return "Baseline mortal performance"
    def predict_future(self):
        if not self.avg_bat_speed: return {}
        peak_age = 27.5
        current_age = 25
        age_factor = max(0.8, 1 - abs(current_age - peak_age)/10)
        return {
            'next_10_swings': {
                'hits': int((self.contact or 0) * 0.3 * age_factor),
                'blasts': int((self.blast_per_swing or 0) * 10),
                'whiffs': int((self.whiff_per_swing or 0) * 10)
            },
            'peak_age': peak_age,
            'decline_start': peak_age + 4.5
        }

# --- Ingestion & Feature Engineering ---
def add_delta_and_oscillator(df, col):
    df[f'delta_{col}'] = df[col].diff()
    df[f'oscillator_{col}'] = (df[col] - df[col].rolling(5).mean()) / (df[col].rolling(5).std() + 1e-6)
    return df

def engineer_features(df):
    for col in ['avg_bat_speed', 'batter_run_value']:
        if col in df.columns:
            df = add_delta_and_oscillator(df, col)
    if 'oscillator_avg_bat_speed' in df.columns:
        df['cashout_signal'] = df['oscillator_avg_bat_speed'] < -2
        df['pick_tracked'] = True
    return df

@app.route('/upload_stats', methods=['POST'])
def upload_stats():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    files = request.files.getlist('files')
    all_alerts = []
    results = []
    session = db.session
    try:
        for file_storage in files:
            filename = secure_filename(file_storage.filename)
            if filename.endswith('.zip'):
                try:
                    with zipfile.ZipFile(file_storage, 'r') as zipf:
                        for zipinfo in zipf.infolist():
                            if zipinfo.filename.endswith('.csv'):
                                _process_csv(zipf.open(zipinfo), results, all_alerts)
                except zipfile.BadZipFile:
                    all_alerts.append(f"Corrupted ZIP file: {filename} could not be opened.")
                    continue
            elif filename.endswith('.csv'):
                _process_csv(file_storage, results, all_alerts)
        session.commit()
        return jsonify({'status': 'INGESTION COMPLETE', 'alerts': all_alerts, 'results': results})
    except Exception as e:
        session.rollback()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

def _process_csv(file_obj, results, all_alerts):
    try:
        df = pd.read_csv(file_obj)
        df = engineer_features(df)
        for _, row in df.iterrows():
            omni = Omniscience(
                **{col: row.get(col) for col in Omniscience.__table__.columns.keys() if col in row},
                delta_bat_speed=row.get('delta_avg_bat_speed'),
                oscillator_bat_speed=row.get('oscillator_avg_bat_speed'),
                cashout_signal=row.get('cashout_signal', False),
                pick_tracked=row.get('pick_tracked', False),
                timestamp=datetime.utcnow()
            )
            db.session.add(omni)
            results.append({'name': omni.name, 'cashout_signal': omni.cashout_signal, 'oscillator_bat_speed': omni.oscillator_bat_speed})
    except Exception as e:
        all_alerts.append(f"Error processing CSV: {str(e)}")

# --- Stats API ---
@app.route('/api/omniscience_stats', methods=['GET'])
def omniscience_stats():
    stats = Omniscience.query.order_by(Omniscience.timestamp.desc()).limit(100).all()
    return jsonify([s.to_prophecy() for s in stats])

# --- Dashboard Example ---
@app.route('/divine_dashboard')
def divine_dashboard():
    stats = Omniscience.query.order_by(Omniscience.timestamp.desc()).limit(20).all()
    return render_template_string("""
    <html><head><title>Omniscient Dashboard</title></head>
    <body style='background:#181818;color:#FFD700;font-family:sans-serif'>
    <h1>Omniscient Sports Intelligence Dashboard</h1>
    <table border=1 cellpadding=6>
    <tr>
      <th>Name</th><th>Avg Bat Speed</th><th>Delta Bat Speed</th>
      <th>Oscillator</th><th>Cashout Signal</th><th>Divine Insight</th>
    </tr>
    {% for s in stats %}
      <tr>
        <td>{{s.name}}</td>
        <td>{{s.avg_bat_speed}}</td>
        <td>{{s.delta_bat_speed}}</td>
        <td>{{s.oscillator_bat_speed}}</td>
        <td style="color:{{'red' if s.cashout_signal else 'green'}}">{{s.cashout_signal}}</td>
        <td>{{s.generate_divine_insight()}}</td>
      </tr>
    {% endfor %}
    </table>
    </body></html>
    """, stats=stats)

@app.route('/')
def index():
    return jsonify({
        'message': 'OMNISCIENT SPORTS INTELLIGENCE SYSTEM',
        'status': 'GOD MODE',
        'endpoints': {
            'upload': '/upload_stats [POST]',
            'stats': '/api/omniscience_stats [GET]',
            'dashboard': '/divine_dashboard [GET]'
        }
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
