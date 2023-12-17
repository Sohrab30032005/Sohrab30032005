from flask import Flask, request, jsonify
from datetime import datetime, timedelta

app = Flask(__name__)

# Simulated database of tablets with expiry dates
tablets = [
    {"id": 1, "name": "Tablet A", "expiry_date": "2023-12-31"},
    {"id": 2, "name": "Tablet B", "expiry_date": "2023-11-15"},
    # Add more tablets as needed
]

# Endpoint to get tablets nearing expiry
@app.route('/api/tablets/expiry_alert', methods=['GET'])
def expiry_alert():
    today = datetime.now().date()
    nearing_expiry = []

    for tablet in tablets:
        expiry_date = datetime.strptime(tablet['expiry_date'], "%Y-%m-%d").date()
        days_until_expiry = (expiry_date - today).days

        if days_until_expiry <= 30 and days_until_expiry >= 0:
            nearing_expiry.append({
                "tablet_id": tablet['id'],
                "tablet_name": tablet['name'],
                "expiry_date": tablet['expiry_date'],
                "days_until_expiry": days_until_expiry
            })

    return jsonify(nearing_expiry)

if __name__ == '__main__':
    app.run(debug=True)
