1| import os
2| import io
3| import csv
4| import zipfile
5| import tempfile
6| import pandas as pd
7| import numpy as np
8| from datetime import datetime
9| from flask import Flask, request, jsonify, render_template_string, abort
10| from flask_sqlalchemy import SQLAlchemy
11| from werkzeug.utils import secure_filename
12| 
13| app = Flask(__name__)
14| app.secret_key = os.getenv('SECRET_KEY', 'omniscient-divine-key')
15| app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///omniscience.db')
16| app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
17| app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
18| app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
19| db = SQLAlchemy(app)
20| 
21| # --- Models ---
22| class BaseModel:
23|     def to_dict(self):
24|         return {c.name: getattr(self, c.name) for c in self.__table__.columns}
25|     def to_prophecy(self):
26|         d = self.to_dict()
27|         d['divine_insight'] = self.generate_divine_insight()
28|         d['future_prediction'] = self.predict_future()
29|         return d
30| 
31| class NBAStat(db.Model, BaseModel):
32|     id = db.Column(db.Integer, primary_key=True)
33|     player_id = db.Column(db.String(50), nullable=False, index=True)
34|     name = db.Column(db.String(100), nullable=False)
35|     team = db.Column(db.String(50))
36|     position = db.Column(db.String(20))
37|     points = db.Column(db.Float)
38|     rebounds = db.Column(db.Float)
39|     assists = db.Column(db.Float)
40|     steals = db.Column(db.Float)
41|     blocks = db.Column(db.Float)
42|     turnovers = db.Column(db.Float)
43|     minutes_played = db.Column(db.Float)
44|     season = db.Column(db.String(20), nullable=False, index=True)
45|     divine_score = db.Column(db.Float)
46|     prophecy_rating = db.Column(db.Float)
47|     future_value = db.Column(db.Float)
48|     def generate_divine_insight(self):
49|         insights = []
50|         if self.points and self.points > 25:
51|             insights.append(f"{self.name} is a scoring deity - {self.points} PPG")
52|         if self.assists and self.assists > 8:
53|             insights.append(f"Vision of a playmaking god - {self.assists} APG")
54|         if self.steals and self.blocks and self.steals + self.blocks > 3:
55|             insights.append(f"Defensive omnipotence - {self.steals+self.blocks} combined steals/blocks")
56|         return insights or ["Mortal performance"]
57|     def predict_future(self):
58|         if not self.minutes_played: return "Unknown future"
59|         growth = 1.1 if float(self.season[:4]) < 2023 else 0.98
60|         return {
61|             'next_season_points': round((self.points or 0) * growth, 1),
62|             'peak_season': int(self.season[:4]) + 2,
63|             'decline_age': 32 if self.position in ['PG', 'SG'] else 34
64|         }
65| 
66| class Omniscience(db.Model, BaseModel):
67|     id = db.Column(db.Integer, primary_key=True)
68|     name = db.Column(db.String(100), nullable=False, index=True)
69|     swings_competitive = db.Column(db.Integer)
70|     percent_swings_competitive = db.Column(db.Float)
71|     contact = db.Column(db.Integer)
72|     avg_bat_speed = db.Column(db.Float)
73|     hard_swing_rate = db.Column(db.Float)
74|     squared_up_per_bat_contact = db.Column(db.Float)
75|     squared_up_per_swing = db.Column(db.Float)
76|     blast_per_bat_contact = db.Column(db.Float)
77|     blast_per_swing = db.Column(db.Float)
78|     swing_length = db.Column(db.Float)
79|     swords = db.Column(db.Integer)
80|     batter_run_value = db.Column(db.Float)
81|     whiffs = db.Column(db.Integer)
82|     whiff_per_swing = db.Column(db.Float)
83|     batted_ball_events = db.Column(db.Integer)
84|     batted_ball_event_per_swing = db.Column(db.Float)
85|     timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
86|     cashout_signal = db.Column(db.Boolean, default=False)
87|     pick_tracked = db.Column(db.Boolean, default=False)
88|     delta_bat_speed = db.Column(db.Float)
89|     oscillator_bat_speed = db.Column(db.Float)
90|     prophecy_rating = db.Column(db.Float)
91|     future_value = db.Column(db.Float)
92|     def generate_divine_insight(self):
93|         if self.oscillator_bat_speed and self.oscillator_bat_speed < -2.5:
94|             return "DIVINE INTERVENTION: Cashout signal STRONG - bat speed in critical decline"
95|         if self.blast_per_swing and self.blast_per_swing > 0.35:
96|             return "Olympian power: Elite blast rate detected"
97|         if self.whiff_per_swing and self.whiff_per_swing > 0.4:
98|             return "Mortal weakness: High whiff rate vulnerable to divine pitches"
99|         return "Baseline mortal performance"
100|     def predict_future(self):
101|         if not self.avg_bat_speed: return {}
102|         peak_age = 27.5
103|         current_age = 25
104|         age_factor = max(0.8, 1 - abs(current_age - peak_age)/10)
105|         return {
106|             'next_10_swings': {
107|                 'hits': int((self.contact or 0) * 0.3 * age_factor),
108|                 'blasts': int((self.blast_per_swing or 0) * 10),
109|                 'whiffs': int((self.whiff_per_swing or 0) * 10)
110|             },
111|             'peak_age': peak_age,
112|             'decline_start': peak_age + 4.5
113|         }
114| 
115| # --- Ingestion & Feature Engineering ---
116| def add_delta_and_oscillator(df, col):
117|     df[f'delta_{col}'] = df[col].diff()
118|     df[f'oscillator_{col}'] = (df[col] - df[col].rolling(5).mean()) / (df[col].rolling(5).std() + 1e-6)
119|     return df
120| 
121| def engineer_features(df):
122|     for col in ['avg_bat_speed', 'batter_run_value']:
123|         if col in df.columns:
124|             df = add_delta_and_oscillator(df, col)
125|     if 'oscillator_avg_bat_speed' in df.columns:
126|         df['cashout_signal'] = df['oscillator_avg_bat_speed'] < -2
127|         df['pick_tracked'] = True
128|     return df
129| 
130| # --- Utilities ---
131| def is_csv_corrupt(file_obj):
132|     try:
133|         df = pd.read_csv(file_obj)
134|         file_obj.seek(0)
135|         return False, df
136|     except Exception as e:
137|         return True, str(e)
138| 
139| def is_zip_corrupt(file_obj):
140|     try:
141|         with zipfile.ZipFile(file_obj) as z:
142|             csv_files = [name for name in z.namelist() if name.endswith('.csv')]
143|             if not csv_files:
144|                 return True, "No CSV files found in ZIP."
145|             with z.open(csv_files[0]) as f:
146|                 try:
147|                     df = pd.read_csv(f)
148|                     return False, df
149|                 except Exception as e:
150|                     return True, f"Corrupt CSV in ZIP: {str(e)}"
151|     except zipfile.BadZipFile:
152|         return True, "ZIP file is corrupted or invalid."
153|     except Exception as e:
154|         return True, str(e)
155| 
156| # --- CSV processor ---
157| def _process_csv(file_obj, results, all_alerts):
158|     corrupt, df_or_err = is_csv_corrupt(file_obj)
159|     if corrupt:
160|         all_alerts.append(f"Error processing CSV: {df_or_err}")
161|         return
162|     df = engineer_features(df_or_err)
163|     for _, row in df.iterrows():
164|         omni = Omniscience(
165|             **{col: row.get(col) for col in Omniscience.__table__.columns.keys() if col in row},
166|             delta_bat_speed=row.get('delta_avg_bat_speed'),
167|             oscillator_bat_speed=row.get('oscillator_avg_bat_speed'),
168|             cashout_signal=row.get('cashout_signal', False),
169|             pick_tracked=row.get('pick_tracked', False),
170|             timestamp=datetime.utcnow()
171|         )
172|         db.session.add(omni)
173|         results.append({'name': omni.name, 'cashout_signal': omni.cashout_signal, 'oscillator_bat_speed': omni.oscillator_bat_speed})
174| 
175| # --- File Upload Route ---
176| @app.route('/upload_stats', methods=['POST'])
177| def upload_stats():
178|     if 'files' not in request.files:
179|         return jsonify({'error': 'No files provided'}), 400
180|     files = request.files.getlist('files')
181|     all_alerts = []
182|     results = []
183|     session = db.session
184|     try:
185|         for file_storage in files:
186|             filename = secure_filename(file_storage.filename)
187|             if filename.endswith('.zip'):
188|                 try:
189|                     with zipfile.ZipFile(file_storage, 'r') as zipf:
190|                         for zipinfo in zipf.infolist():
191|                             if zipinfo.filename.endswith('.csv'):
192|                                 _process_csv(zipf.open(zipinfo), results, all_alerts)
193|                 except zipfile.BadZipFile:
194|                     all_alerts.append(f"Corrupted ZIP file: {filename} could not be opened.")
195|                     continue
196|             elif filename.endswith('.csv'):
197|                 _process_csv(file_storage, results, all_alerts)
198|         session.commit()
199|         return jsonify({'status': 'INGESTION COMPLETE', 'alerts': all_alerts, 'results': results})
200|     except Exception as e:
201|         session.rollback()
202|         return jsonify({'error': f'Server error: {str(e)}'}), 500
203| 
204| # --- API Endpoints ---
205| @app.route('/api/omniscience_stats', methods=['GET'])
206| def omniscience_stats():
207|     stats = Omniscience.query.order_by(Omniscience.timestamp.desc()).limit(100).all()
208|     return jsonify([s.to_prophecy() for s in stats])
209| 
210| @app.route('/divine_dashboard')
211| def divine_dashboard():
212|     stats = Omniscience.query.order_by(Omniscience.timestamp.desc()).limit(20).all()
213|     return render_template_string("""
214|     <html><head><title>Omniscient Dashboard</title></head>
215|     <body style='background:#181818;color:#FFD700;font-family:sans-serif'>
216|     <h1>Omniscient Sports Intelligence Dashboard</h1>
217|     <table border=1 cellpadding=6>
218|     <tr>
219|       <th>Name</th><th>Avg Bat Speed</th><th>Delta Bat Speed</th>
220|       <th>Oscillator</th><th>Cashout Signal</th><th>Divine Insight</th>
221|     </tr>
222|     {% for s in stats %}
223|       <tr>
224|         <td>{{s.name}}</td>
225|         <td>{{s.avg_bat_speed}}</td>
226|         <td>{{s.delta_bat_speed}}</td>
227|         <td>{{s.oscillator_bat_speed}}</td>
228|         <td style="color:{{'red' if s.cashout_signal else 'green'}}">{{s.cashout_signal}}</td>
229|         <td>{{s.generate_divine_insight()}}</td>
230|       </tr>
231|     {% endfor %}
232|     </table>
233|     </body></html>
234|     """, stats=stats)
235| 
236| @app.route('/')
237| def index():
238|     return jsonify({
239|         'message': 'OMNISCIENT SPORTS INTELLIGENCE SYSTEM',
240|         'status': 'GOD MODE',
241|         'endpoints': {
242|             'upload': '/upload_stats [POST]',
243|             'stats': '/api/omniscience_stats [GET]',
244|             'dashboard': '/divine_dashboard [GET]'
245|         }
246|     })
247| 
248| if __name__ == '__main__':
249|     with app.app_context():
250|         db.create_all()
251|     app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
