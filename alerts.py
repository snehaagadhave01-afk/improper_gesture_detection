"""
alerts.py
---------
Sends an instant email alert for high-severity incidents (e.g. improper
gesture, person not visible) -- addresses "lacks an automatic alert
mechanism."

Off by default. To enable:
  1. In config.py set ENABLE_EMAIL_ALERTS = True
  2. Fill in SMTP_SENDER_EMAIL / SMTP_SENDER_APP_PASSWORD / ALERT_RECEIVER_EMAIL

For Gmail: generate an "app password" at
https://myaccount.google.com/apppasswords (your normal password won't
work). Any SMTP provider works if you change SMTP_SERVER/SMTP_PORT.

NOTE: SMS alerts aren't included here -- that needs a paid third-party
service (e.g. Twilio) with its own account/API keys. If you want that,
it's a small addition on top of this once you have Twilio credentials.
"""

import smtplib
import time
from email.mime.text import MIMEText

import config

_last_sent_time = 0.0


def maybe_send_email_alert(incident_type: str, message: str):
    global _last_sent_time

    if not config.ENABLE_EMAIL_ALERTS:
        return
    if incident_type not in config.EMAIL_ALERT_TYPES:
        return

    now = time.time()
    if now - _last_sent_time < config.EMAIL_ALERT_COOLDOWN_SECONDS:
        return

    try:
        msg = MIMEText(f"{message}\n\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        msg["Subject"] = f"[Monitoring Alert] {incident_type}"
        msg["From"] = config.SMTP_SENDER_EMAIL
        msg["To"] = config.ALERT_RECEIVER_EMAIL

        with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
            server.starttls()
            server.login(config.SMTP_SENDER_EMAIL, config.SMTP_SENDER_APP_PASSWORD)
            server.send_message(msg)

        _last_sent_time = now
        print(f"[EMAIL] Alert sent: {incident_type}")
    except Exception as e:
        print(f"[EMAIL] Failed to send alert: {e}")