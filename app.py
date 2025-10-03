from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import json
import logging
import os
import time
from datetime import datetime
import threading
import requests
import sqlite3
from dotenv import load_dotenv
import numpy as np
from sklearn.ensemble import IsolationForest
import cv2

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='suraksha_server.log'
)
logger = logging.getLogger('suraksha_server')

# Create Flask app and SocketIO instance
app = Flask(__name__, 
            static_folder='static',
            static_url_path='')
app.config['SECRET_KEY'] = 'suraksha_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration from environment variables
PORT = int(os.getenv('PORT', 3000))
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
DATABASE_FILE = os.getenv('DATABASE_FILE', 'suraksha.db')
ACC_THRESHOLD = float(os.getenv('ACC_THRESHOLD', 15.0))
GYRO_THRESHOLD = float(os.getenv('GYRO_THRESHOLD', 45.0))
IMPACT_DURATION_MS = int(os.getenv('IMPACT_DURATION_MS', 200))
CONSECUTIVE_DETECTIONS_REQUIRED = int(os.getenv('CONSECUTIVE_DETECTIONS_REQUIRED', 3))
VIDEO_CALL_TIMEOUT_SECONDS = int(os.getenv('VIDEO_CALL_TIMEOUT_SECONDS', 60))

# Emergency contacts from environment variables
EMERGENCY_CONTACTS = [
    {
        "name": os.getenv('EMERGENCY_CONTACT_1_NAME', 'Emergency Contact 1'),
        "phone": os.getenv('EMERGENCY_CONTACT_1_PHONE', '+1234567890'),
        "email": os.getenv('EMERGENCY_CONTACT_1_EMAIL', 'contact1@example.com')
    },
    {
        "name": os.getenv('EMERGENCY_CONTACT_2_NAME', 'Emergency Contact 2'),
        "phone": os.getenv('EMERGENCY_CONTACT_2_PHONE', '+0987654321'),
        "email": os.getenv('EMERGENCY_CONTACT_2_EMAIL', 'contact2@example.com')
    }
]

# Emergency services from environment variables
EMERGENCY_SERVICES = {
    "ambulance": os.getenv('EMERGENCY_SERVICE_AMBULANCE', '+1122334455'),
    "police": os.getenv('EMERGENCY_SERVICE_POLICE', '+5566778899')
}

# Twilio configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', '')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER', '')

# Active accident sessions
active_accidents = {}

# Machine learning model for accident detection
model = None

def train_anomaly_detection_model(data_samples=1000):
    """
    Train an anomaly detection model using Isolation Forest
    """
    global model
    
    # Generate synthetic normal driving data
    np.random.seed(42)
    normal_acc_x = np.random.normal(0, 3, data_samples)
    normal_acc_y = np.random.normal(0, 3, data_samples)
    normal_acc_z = np.random.normal(9.8, 1, data_samples)  # Earth's gravity
    normal_gyro_x = np.random.normal(0, 10, data_samples)
    normal_gyro_y = np.random.normal(0, 10, data_samples)
    normal_gyro_z = np.random.normal(0, 10, data_samples)
    
    # Combine features
    X = np.column_stack([
        normal_acc_x, normal_acc_y, normal_acc_z,
        normal_gyro_x, normal_gyro_y, normal_gyro_z
    ])
    
    # Train isolation forest model
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)
    
    logger.info("Anomaly detection model trained successfully")

# Initialize database
def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS accidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        acceleration_x REAL,
        acceleration_y REAL,
        acceleration_z REAL,
        gyroscope_x REAL,
        gyroscope_y REAL,
        gyroscope_z REAL,
        latitude REAL,
        longitude REAL,
        confirmed INTEGER DEFAULT 0,
        override INTEGER DEFAULT 0,
        emergency_notified INTEGER DEFAULT 0,
        video_call_session TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sensor_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        acceleration_x REAL,
        acceleration_y REAL,
        acceleration_z REAL,
        gyroscope_x REAL,
        gyroscope_y REAL,
        gyroscope_z REAL,
        latitude REAL,
        longitude REAL,
        anomaly_score REAL
    )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized")

# Store accident data in database
def store_accident(data):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Extract data from the JSON
    timestamp = data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    acc = data.get('acceleration', {})
    gyro = data.get('gyroscope', {})
    loc = data.get('location', {})
    
    cursor.execute('''
    INSERT INTO accidents 
    (timestamp, acceleration_x, acceleration_y, acceleration_z, 
     gyroscope_x, gyroscope_y, gyroscope_z, 
     latitude, longitude)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp,
        acc.get('x', 0.0), acc.get('y', 0.0), acc.get('z', 0.0),
        gyro.get('x', 0.0), gyro.get('y', 0.0), gyro.get('z', 0.0),
        loc.get('latitude', 0.0), loc.get('longitude', 0.0)
    ))
    
    accident_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    logger.info(f"Accident data stored with ID: {accident_id}")
    return accident_id

# Update accident record
def update_accident(accident_id, confirmed=None, override=None, emergency_notified=None, video_call_session=None):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    update_parts = []
    params = []
    
    if confirmed is not None:
        update_parts.append("confirmed = ?")
        params.append(1 if confirmed else 0)
    
    if override is not None:
        update_parts.append("override = ?")
        params.append(1 if override else 0)
    
    if emergency_notified is not None:
        update_parts.append("emergency_notified = ?")
        params.append(1 if emergency_notified else 0)
        
    if video_call_session is not None:
        update_parts.append("video_call_session = ?")
        params.append(video_call_session)
    
    if update_parts:
        query = f"UPDATE accidents SET {', '.join(update_parts)} WHERE id = ?"
        params.append(accident_id)
        cursor.execute(query, params)
        conn.commit()
    
    conn.close()
    logger.info(f"Accident ID {accident_id} updated")

# Log sensor data for analysis
def log_sensor_data(data, anomaly_score=0):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Extract data from the JSON
    timestamp = data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    acc = data.get('acceleration', {})
    gyro = data.get('gyroscope', {})
    loc = data.get('location', {})
    
    cursor.execute('''
    INSERT INTO sensor_logs 
    (timestamp, acceleration_x, acceleration_y, acceleration_z, 
     gyroscope_x, gyroscope_y, gyroscope_z, 
     latitude, longitude, anomaly_score)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp,
        acc.get('x', 0.0), acc.get('y', 0.0), acc.get('z', 0.0),
        gyro.get('x', 0.0), gyro.get('y', 0.0), gyro.get('z', 0.0),
        loc.get('latitude', 0.0), loc.get('longitude', 0.0),
        anomaly_score
    ))
    
    conn.commit()
    conn.close()

# Detect accident using machine learning
def ml_detect_accident(data):
    global model
    
    if model is None:
        train_anomaly_detection_model()
    
    # Extract features
    acc = data.get('acceleration', {})
    gyro = data.get('gyroscope', {})
    
    features = np.array([
        acc.get('x', 0.0), 
        acc.get('y', 0.0), 
        acc.get('z', 0.0),
        gyro.get('x', 0.0), 
        gyro.get('y', 0.0), 
        gyro.get('z', 0.0)
    ]).reshape(1, -1)
    
    # Get anomaly score (-1 for anomalies, 1 for normal)
    prediction = model.predict(features)[0]
    score = model.score_samples(features)[0]
    
    # Log data with anomaly score
    log_sensor_data(data, score)
    
    # Return True if anomaly (accident) detected
    return prediction == -1

# Traditional threshold-based accident detection
def threshold_detect_accident(data):
    acc = data.get('acceleration', {})
    gyro = data.get('gyroscope', {})
    
    # Calculate magnitudes
    acc_magnitude = np.sqrt(
        acc.get('x', 0.0)**2 + 
        acc.get('y', 0.0)**2 + 
        acc.get('z', 0.0)**2
    )
    
    gyro_magnitude = np.sqrt(
        gyro.get('x', 0.0)**2 + 
        gyro.get('y', 0.0)**2 + 
        gyro.get('z', 0.0)**2
    )
    
    # Check if thresholds are exceeded
    return acc_magnitude > ACC_THRESHOLD and gyro_magnitude > GYRO_THRESHOLD

# Combined accident detection (fusion of ML and threshold-based)
def detect_accident(data):
    # Use both methods and combine results
    ml_detection = ml_detect_accident(data)
    threshold_detection = threshold_detect_accident(data)
    
    # If either method detects an accident, consider it detected
    # In a production system, you might want to weight these differently
    return ml_detection or threshold_detection

# Notify emergency services
def notify_emergency_services(accident_data, accident_id):
    logger.info("Notifying emergency services...")
    
    # In a real implementation, integrate with SMS/call APIs
    # For example, using Twilio:
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
        try:
            from twilio.rest import Client
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            
            location = accident_data.get('location', {})
            lat = location.get('latitude', 'unknown')
            lon = location.get('longitude', 'unknown')
            message_body = f"EMERGENCY: Accident detected at coordinates: {lat}, {lon}"
            
            # Notify emergency contacts
            for contact in EMERGENCY_CONTACTS:
                if contact['phone']:
                    message = client.messages.create(
                        body=message_body,
                        from_=TWILIO_PHONE_NUMBER,
                        to=contact['phone']
                    )
                    logger.info(f"Twilio message sent to {contact['name']}: {message.sid}")
        except Exception as e:
            logger.error(f"Error sending Twilio notifications: {e}")
    else:
        # Simulate notification
        location = accident_data.get('location', {})
        lat = location.get('latitude', 'unknown')
        lon = location.get('longitude', 'unknown')
        
        message = f"EMERGENCY: Accident detected at coordinates: {lat}, {lon}"
        logger.info(f"Emergency message: {message}")
        
        # Notify all emergency contacts
        for contact in EMERGENCY_CONTACTS:
            logger.info(f"Notifying {contact['name']} at {contact['phone']}")
        
        # Notify emergency services
        for service, number in EMERGENCY_SERVICES.items():
            logger.info(f"Notifying {service} at {number}")
    
    # Update database to record that notifications were sent
    update_accident(accident_id, emergency_notified=True)
    
    return True

# Initiate video call
def initiate_video_call(accident_data, accident_id):
    """
    Initiate a WebRTC video call session for accident verification.
    Returns a unique session ID for the video call.
    """
    session_id = f"accident_{accident_id}_{int(time.time())}"
    
    # Store session information
    active_accidents[session_id] = {
        'accident_id': accident_id,
        'data': accident_data,
        'created_at': datetime.now(),
        'status': 'pending',  # pending, confirmed, cancelled
        'participants': []
    }
    
    # Update accident record with session ID
    update_accident(accident_id, video_call_session=session_id)
    
    logger.info(f"Video call initiated with session ID: {session_id}")
    
    # Notify clients via WebSocket that a new call is available
    socketio.emit('new_accident', {
        'session_id': session_id,
        'location': accident_data.get('location', {}),
        'timestamp': accident_data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    })
    
    # Start a timer for the video call timeout
    threading.Thread(target=video_call_timeout, args=(session_id, accident_id, accident_data)).start()
    
    return session_id

# Handle video call timeout
def video_call_timeout(session_id, accident_id, accident_data):
    """
    Wait for the video call timeout period, then check if the call was answered.
    If not, automatically notify emergency services.
    """
    time.sleep(VIDEO_CALL_TIMEOUT_SECONDS)
    
    if session_id in active_accidents and active_accidents[session_id]['status'] == 'pending':
        # No response, assume emergency and notify
        logger.info(f"No response for session {session_id}, notifying emergency services")
        notify_emergency_services(accident_data, accident_id)
        
        # Update session status
        active_accidents[session_id]['status'] = 'confirmed'
        update_accident(accident_id, confirmed=True)

# Flask routes
@app.route('/')
def index():
    """Serve the main dashboard page"""
    return render_template('index.html')

@app.route('/video-call/<session_id>')
def video_call(session_id):
    """Serve the video call page"""
    if session_id not in active_accidents:
        return jsonify({"error": "Invalid session ID"}), 404
    
    return render_template('video_call.html', session_id=session_id)

@app.route('/accident', methods=['POST'])
def accident_handler():
    """
    Endpoint to handle accident alerts.
    Expects JSON with fields like 'acceleration', 'gyroscope', and 'location'.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Check if this is an override request
    if data.get('override'):
        session_id = data.get('session_id')
        if session_id and session_id in active_accidents:
            # Handle override (cancel alert)
            logger.info(f"Received override request for session {session_id}")
            
            accident_id = active_accidents[session_id]['accident_id']
            active_accidents[session_id]['status'] = 'cancelled'
            
            # Update accident record
            update_accident(accident_id, override=True)
            
            # Notify clients via WebSocket
            socketio.emit('accident_override', {
                'session_id': session_id,
                'message': 'Accident alert cancelled by user'
            })
            
            return jsonify({"message": "Override received, alert cancelled"}), 200
        else:
            return jsonify({"error": "Invalid session ID for override"}), 400

    # Check if required fields are present
    if 'acceleration' not in data or 'location' not in data:
        return jsonify({"error": "Missing required fields: 'acceleration' and 'location'"}), 400

    # Log the received data
    logger.info(f"Received sensor data: {data}")
    
    # Check for accident
    is_accident = detect_accident(data)
    
    if is_accident:
        logger.info("Potential accident detected!")
        
        # Store in database
        accident_id = store_accident(data)
        
        # Initiate a video call to confirm the accident
        session_id = initiate_video_call(data, accident_id)
        
        return jsonify({
            "message": "Accident detected, initiating video call",
            "accident_id": accident_id,
            "session_id": session_id
        }), 200
    else:
        # Just log the data, no accident detected
        return jsonify({"message": "Data received, no accident detected"}), 200

@app.route('/confirm-accident', methods=['POST'])
def confirm_accident():
    """
    Endpoint to confirm an accident after video call verification.
    """
    data = request.get_json()
    if not data or 'session_id' not in data:
        return jsonify({"error": "Missing session ID"}), 400
    
    session_id = data['session_id']
    
    if session_id not in active_accidents:
        return jsonify({"error": "Invalid session ID"}), 404
    
    accident_id = active_accidents[session_id]['accident_id']
    accident_data = active_accidents[session_id]['data']
    
    # Update session and accident status
    active_accidents[session_id]['status'] = 'confirmed'
    update_accident(accident_id, confirmed=True)
    
    # Notify emergency services
    notify_emergency_services(accident_data, accident_id)
    
    # Notify clients via WebSocket
    socketio.emit('accident_confirmed', {
        'session_id': session_id,
        'message': 'Accident confirmed, notifying emergency services'
    })
    
    return jsonify({"message": "Accident confirmed, emergency services notified"}), 200

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('join_video_call')
def handle_join_video_call(data):
    session_id = data.get('session_id')
    
    if not session_id or session_id not in active_accidents:
        emit('error', {'message': 'Invalid session ID'})
        return
    
    # Add client to the session participants
    active_accidents[session_id]['participants'].append(request.sid)
    
    # Join the room for this session
    from flask_socketio import join_room
    join_room(session_id)
    
    # Notify others in the room
    emit('user_joined', {'sid': request.sid}, room=session_id, include_self=False)
    
    logger.info(f"User {request.sid} joined video call {session_id}")

@socketio.on('ice_candidate')
def handle_ice_candidate(data):
    session_id = data.get('session_id')
    candidate = data.get('candidate')
    target_sid = data.get('target_sid')
    
    if session_id and session_id in active_accidents and candidate:
        if target_sid:
            # Send to specific user
            emit('ice_candidate', {
                'candidate': candidate,
                'from_sid': request.sid
            }, room=target_sid)
        else:
            # Broadcast to all in the room except sender
            emit('ice_candidate', {
                'candidate': candidate,
                'from_sid': request.sid
            }, room=session_id, include_self=False)

@socketio.on('offer')
def handle_offer(data):
    session_id = data.get('session_id')
    offer = data.get('offer')
    target_sid = data.get('target_sid')
    
    if session_id and session_id in active_accidents and offer and target_sid:
        emit('offer', {
            'offer': offer,
            'from_sid': request.sid
        }, room=target_sid)

@socketio.on('answer')
def handle_answer(data):
    session_id = data.get('session_id')
    answer = data.get('answer')
    target_sid = data.get('target_sid')
    
    if session_id and session_id in active_accidents and answer and target_sid:
        emit('answer', {
            'answer': answer,
            'from_sid': request.sid
        }, room=target_sid)

# Initialize the application
def initialize_app():
    # Initialize database
    init_db()
    
    # Train ML model
    train_anomaly_detection_model()
    
    logger.info("Suraksha Accident Detection System initialized")

if __name__ == '__main__':
    # Initialize the application
    initialize_app()
    
    # Run the Flask server
    logger.info(f"Starting server on port {PORT}")
    socketio.run(app, debug=DEBUG, host='0.0.0.0', port=PORT)
