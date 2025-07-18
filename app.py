import os
import pickle
from dotenv import load_dotenv
from groq import Groq
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import threading
import uuid
from datetime import datetime

# Charger les variables d'environnement
load_dotenv()
# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Chemin vers les artefacts ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(CURRENT_DIR, "fraud_detection_artifacts")

# Variables globales
transactions = []
# Chargement des artefacts
def load_artifact(filename):
    with open(os.path.join(ARTIFACTS_DIR, filename), 'rb') as f:
        return pickle.load(f)

imputer_num = load_artifact("imputer_num.pkl")
imputer_cat = load_artifact("imputer_cat.pkl")
scaler = load_artifact("scaler.pkl")
encoders = load_artifact("encoders.pkl")
model = load_artifact("model_lightgbm.pkl")

# Colonnes utilisées (doivent correspondre à l'entraînement)
categorical_cols = list(encoders.keys())
numerical_cols = list(imputer_num.feature_names_in_) if hasattr(imputer_num, 'feature_names_in_') else []

# Fonction de prétraitement
def preprocess_data(df):
    df = df.copy()
    # S'assurer que toutes les colonnes sont présentes
    for col in numerical_cols:
        if col not in df.columns:
            df[col] = np.nan
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "missing"
    # Imputation
    if numerical_cols:
        df[numerical_cols] = imputer_num.transform(df[numerical_cols])
    if categorical_cols:
        df[categorical_cols] = imputer_cat.transform(df[categorical_cols])
    # Standardisation
    if numerical_cols:
        df[numerical_cols] = scaler.transform(df[numerical_cols])
    # Encodage
    for col in categorical_cols:
        df[col] = df[col].astype(str).map(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])
        df[col] = encoders[col].transform(df[col])
    # Réindexer pour l'ordre des features du modèle
    model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else df.columns
    df = df.reindex(columns=model_features, fill_value=0)
    return df
# Initialiser le client Groq
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Prompt système pour se concentrer sur la fraude bancaire
SYSTEM_PROMPT = """Tu es un assistant spécialisé dans la fraude bancaire en ligne. 
Tu ne dois répondre qu'aux questions liées à :
- La fraude bancaire en ligne
- Les cyberattaques bancaires
- La sécurité des transactions en ligne
- Les méthodes de protection contre la fraude
- Les types de fraudes bancaires digitales
- La prévention des fraudes en ligne

Si la question n'est pas liée à ces sujets, réponds simplement : "Je ne peux répondre qu'aux questions concernant la fraude bancaire en ligne."

Tes réponses doivent être courtes, précises, concises, informatives et sans style pour le texte.
Tu dois maintenir le contexte de la conversation précédente."""

class FraudChatBot:
    def __init__(self):
        self.sessions = {}  # Dictionnaire pour stocker les sessions
        self.history_dir = "chat_sessions"
        self.max_history_length = 20
        self.lock = threading.Lock()
        
        # Créer le dossier des sessions s'il n'existe pas
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
    
    def get_session_file(self, session_id):
        """Obtenir le chemin du fichier de session"""
        return os.path.join(self.history_dir, f"session_{session_id}.json")
    
    def load_session(self, session_id):
        """Charger l'historique d'une session"""
        session_file = self.get_session_file(session_id)
        try:
            if os.path.exists(session_file):
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('messages', [])
        except Exception as e:
            print(f"Erreur lors du chargement de la session {session_id}: {e}")
        return []
    
    def save_session(self, session_id, conversation_history):
        """Sauvegarder l'historique d'une session"""
        session_file = self.get_session_file(session_id)
        try:
            data = {
                'session_id': session_id,
                'messages': conversation_history[-self.max_history_length:],
                'last_updated': datetime.now().isoformat()
            }
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de la session {session_id}: {e}")
    
    def add_to_session(self, session_id, role, content):
        """Ajouter un message à une session"""
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = self.load_session(session_id)
            
            self.sessions[session_id].append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })
            
            # Maintenir la limite d'historique
            if len(self.sessions[session_id]) > self.max_history_length:
                self.sessions[session_id] = self.sessions[session_id][-self.max_history_length:]
    
    def get_messages_for_api(self, session_id):
        """Préparer les messages pour l'API Groq"""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Obtenir l'historique de la session
        session_history = self.sessions.get(session_id, [])
        
        # Ajouter l'historique récent (sans les timestamps pour l'API)
        for msg in session_history[-10:]:  # Derniers 10 messages
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        return messages
    
    def chat_with_groq(self, user_message, session_id=None):
        """Fonction pour envoyer un message au modèle Groq avec contexte"""
        try:
            # Générer un ID de session si non fourni
            if session_id is None:
                session_id = str(uuid.uuid4())
            
            # Ajouter le message utilisateur à l'historique
            self.add_to_session(session_id, "user", user_message)
            
            # Préparer les messages avec contexte
            messages = self.get_messages_for_api(session_id)
            
            chat_completion = client.chat.completions.create(
                messages=messages,
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.3,
                max_tokens=400,
            )
            
            response = chat_completion.choices[0].message.content
            
            # Ajouter la réponse à l'historique
            self.add_to_session(session_id, "assistant", response)
            
            # Sauvegarder l'historique
            self.save_session(session_id, self.sessions[session_id])
            
            return response, session_id
            
        except Exception as e:
            return f"Erreur lors de la communication avec l'API : {str(e)}", session_id
    
    def get_session_history(self, session_id):
        """Obtenir l'historique d'une session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = self.load_session(session_id)
        return self.sessions.get(session_id, [])
    
    def clear_session(self, session_id):
        """Effacer l'historique d'une session"""
        session_file = self.get_session_file(session_id)
        if session_id in self.sessions:
            del self.sessions[session_id]
        if os.path.exists(session_file):
            os.remove(session_file)

# Initialiser le chatbot
chatbot = FraudChatBot()

# Flask app
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Fraud Detection API (LightGBM, artefacts, full preprocessing) is running!"

@app.route('/predict_transaction', methods=['POST'])
def predict_transaction():
    data = request.get_json()
    df = pd.DataFrame([data])
    df_processed = preprocess_data(df)
    proba = model.predict_proba(df_processed)[0, 1]
    label = "fraud" if proba > 0.5 else "legitimate"
    transactions.append({
        "timestamp": datetime.now().isoformat(),
        "label": label,
        "fraud_score": float(proba),
        "probability": float(proba),
        "input": data  # optionnel, pour garder trace de l'input
    })
    return jsonify({
        "label": label,
        "probability": float(proba),
        "fraud_score": float(proba)
    })

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    data = request.get_json()
    transactions_to_process = data.get("transactions", [])
    df = pd.DataFrame(transactions_to_process)
    df_processed = preprocess_data(df)
    probas = model.predict_proba(df_processed)[:, 1]
    results = []
    for i, proba in enumerate(probas):
        label = "fraud" if proba > 0.5 else "legitimate"
        transaction_record = {
            "timestamp": datetime.now().isoformat(),
            "label": label,
            "fraud_score": float(proba),
            "probability": float(proba),
            "input": transactions_to_process[i]  # ou df.iloc[i].to_dict()
        }
        transactions.append(transaction_record)
        results.append({
            "label": label,
            "probability": float(proba),
            "fraud_score": float(proba)
        })
    return jsonify({"predictions": results})

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 400
    df_processed = preprocess_data(df)
    probas = model.predict_proba(df_processed)[:, 1]
    results = []
    for idx, proba in enumerate(probas):
        label = "fraud" if proba > 0.5 else "legitimate"
        transaction_record = {
            "timestamp": datetime.now().isoformat(),
            "label": label,
            "fraud_score": float(proba),
            "probability": float(proba),
            "input": df.iloc[idx].to_dict()
        }
        transactions.append(transaction_record)
        results.append({
            "row": int(idx),
            "label": label,
            "probability": float(proba),
            "fraud_score": float(proba)
        })
    return jsonify({"predictions": results})

@app.route('/dashboard_stats', methods=['GET'])
def dashboard_stats():
    total = len(transactions)
    frauds = sum(1 for t in transactions if t['label'] == 'fraud')
    legitimate = total - frauds
    fraud_rate = (frauds / total * 100) if total > 0 else 0
    # On retourne les 3 dernières transactions (les plus récentes d'abord)
    recent = transactions[-3:][::-1]

    # Calcul des tendances journalières
    daily = {}
    for t in transactions:
        try:
            dt = datetime.fromisoformat(t['timestamp'])
            key = dt.strftime('%Y-%m-%d')  # Agrégation par jour
        except Exception:
            key = 'unknown'
        if key not in daily:
            daily[key] = {'fraud': 0, 'legitimate': 0}
        if t['label'] == 'fraud':
            daily[key]['fraud'] += 1
        else:
            daily[key]['legitimate'] += 1
    # Trie par date
    dayTrends = [
        {'day': k, 'fraud': v['fraud'], 'legitimate': v['legitimate']}
        for k, v in sorted(daily.items())
    ]

    return jsonify({
        "totalTransactions": total,
        "fraudCount": frauds,
        "legitimateCount": legitimate,
        "fraudRate": fraud_rate,
        "recentTransactions": recent,
        "dayTrends": dayTrends
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint principal pour le chat"""
    try:
        # Obtenir les données JSON
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "Aucune donnée fournie",
                "status": "error"
            }), 400
        
        # Vérifier que la question est fournie
        question = data.get('question')
        if not question:
            return jsonify({
                "error": "Le champ 'question' est requis",
                "status": "error"
            }), 400
        
        # Obtenir l'ID de session (optionnel)
        session_id = data.get('session_id')
        
        # Obtenir la réponse du chatbot
        answer, session_id = chatbot.chat_with_groq(question, session_id)
        
        return jsonify({
            "answer": answer,
            "session_id": session_id,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/chat/history/<session_id>', methods=['GET'])
def get_history(session_id):
    """Endpoint pour obtenir l'historique d'une session"""
    try:
        history = chatbot.get_session_history(session_id)
        return jsonify({
            "history": history,
            "session_id": session_id,
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/chat/clear/<session_id>', methods=['DELETE'])
def clear_history(session_id):
    """Endpoint pour effacer l'historique d'une session"""
    try:
        chatbot.clear_session(session_id)
        return jsonify({
            "message": f"Historique de la session {session_id} effacé",
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de vérification de santé"""
    return jsonify({
        "status": "healthy",
        "service": "Fraud Banking Chat API",
        "timestamp": datetime.now().isoformat()
    })
# Point d'entrée principal
if __name__ == '__main__':
    logger.info("Démarrage de l'API Flask")
    # Démarrer le serveur Flask
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True  # Pour gérer plusieurs requêtes simultanées
    )