import os
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, Response
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from supabase import create_client, Client
from twilio.rest import Client as TwilioClient
from twilio.twiml.messaging_response import MessagingResponse
import logging
import schedule
import google.generativeai as genai
import uuid
import re

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in the environment.")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize Twilio client
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
twilio_client = TwilioClient(account_sid, auth_token)
twilio_whatsapp_number = os.getenv('TWILIO_WHATSAPP_NUMBER')

# Initialize Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    logger.info("Gemini API configured successfully.")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {str(e)}. AI features will be unavailable.")
    gemini_model = None

# Global dictionary to store conversation states
conversations = {}
conversations_lock = threading.Lock()

# Global set to track events being processed (prevents duplicates)
processing_events = set()
processing_lock = threading.Lock()

# --- NOTIFICATION LOGGING ---
def log_notification_to_db(user_id, event_id, notification_type, notification_content, phone_number, twilio_message_sid=None, delivery_status='sent'):
    try:
        notification_data = {
            'user_id': user_id,
            'event_id': event_id,
            'notification_type': notification_type,
            'notification_content': notification_content,
            'phone_number': phone_number,
            'twilio_message_sid': twilio_message_sid,
            'delivery_status': delivery_status,
            'sent_at': datetime.now().isoformat()
        }
        response = supabase.table('notifications_sent').insert(notification_data).execute()
        if response.data:
            logger.info(f"Notification logged to database successfully. ID: {response.data[0]['id']}")
            return response.data[0]['id']
        else:
            logger.warning("Failed to log notification to database - no data returned")
            return None
    except Exception as e:
        logger.error(f"Error logging notification to database: {str(e)}")
        return None

# --- WHATSAPP SENDING ---
def send_whatsapp_notification(phone_number, message, user_id=None, event_id=None, notification_type='event_reminder'):
    try:
        if not phone_number.startswith('+'):
            phone_number = '+' + phone_number
        whatsapp_number = f"whatsapp:{phone_number}"
        logger.info(f"Attempting to send WhatsApp message to {whatsapp_number}")
        twilio_message = twilio_client.messages.create(
            body=message,
            from_=twilio_whatsapp_number,
            to=whatsapp_number
        )
        logger.info(f"WhatsApp message sent successfully to {phone_number}. SID: {twilio_message.sid}")
        if user_id:
            log_notification_to_db(
                user_id=user_id,
                event_id=event_id,
                notification_type=notification_type,
                notification_content=message,
                phone_number=phone_number,
                twilio_message_sid=twilio_message.sid,
                delivery_status='sent'
            )
        return True
    except Exception as e:
        logger.error(f"Failed to send WhatsApp message to {phone_number}: {str(e)}")
        if user_id:
            log_notification_to_db(
                user_id=user_id,
                event_id=event_id,
                notification_type=notification_type,
                notification_content=message,
                phone_number=phone_number,
                twilio_message_sid=None,
                delivery_status='failed'
            )
        return False

# --- SCHEDULER FOR NOTIFICATIONS ---
def check_and_send_notifications():
    execution_id = str(uuid.uuid4())[:8]
    logger.info(f"[{execution_id}] Starting notification check...")
    try:
        today = datetime.now().date()
        logger.info(f"[{execution_id}] Current date for check: {today}")
        response = supabase.table('events').select(
            'id, title, event_date, days_to_notify, event_type, user_id, notified'
        ).or_('notified.is.null,notified.eq.').execute()
        logger.info(f"[{execution_id}] Supabase query returned {len(response.data)} events with notified status NULL/empty.")
        events_to_process = []
        for event in response.data:
            event_date_str = event['event_date']
            event_date = datetime.strptime(event_date_str, '%Y-%m-%d').date()
            days_to_notify = event['days_to_notify']
            notification_date = event_date - timedelta(days=days_to_notify)
            logger.info(f"[{execution_id}] Event '{event['title']}' (ID: {event['id']}): Event Date: {event_date}, Days to Notify: {days_to_notify}, Calculated Notification Date: {notification_date}")
            if today == notification_date:
                events_to_process.append(event)
                logger.info(f"[{execution_id}] MATCH! Event '{event['title']}' is scheduled for notification today.")
            elif today > notification_date:
                supabase.table('events').update({'notified': 'No'}).eq('id', event['id']).execute()
                logger.info(f"[{execution_id}] Event '{event['title']}' notification date passed ({notification_date}), marked as 'No'")
        if not events_to_process:
            logger.info(f"[{execution_id}] No events need notification today after date comparison.")
            return
        notifications_sent = 0
        notifications_failed = 0
        for event in events_to_process:
            event_id = event['id']
            with processing_lock:
                if event_id in processing_events:
                    logger.info(f"[{execution_id}] Event {event_id} is already being processed by another instance. Skipping.")
                    continue
                else:
                    processing_events.add(event_id)
                    logger.info(f"[{execution_id}] Added event {event_id} to processing set.")
            try:
                current_event_response = supabase.table('events').select('notified').eq('id', event_id).single().execute()
                if current_event_response.data and current_event_response.data.get('notified') not in [None, '', 'No']:
                    logger.info(f"[{execution_id}] Event {event_id} has already been notified (status: {current_event_response.data.get('notified')}). Skipping.")
                    continue
                update_response = supabase.table('events').update({'notified': 'Yes'}).eq('id', event_id).execute()
                if update_response.data:
                    logger.info(f"[{execution_id}] Event '{event['title']}' (ID: {event_id}) successfully marked 'Yes' for notification.")
                else:
                    logger.warning(f"[{execution_id}] Event '{event['title']}' (ID: {event_id}) could not be marked 'Yes'. Skipping notification.")
                    notifications_failed += 1
                    continue
                try:
                    user_response = supabase.table('profiles').select('full_name, phone_number').eq('id', event['user_id']).single().execute()
                    user_profile = user_response.data
                except Exception as e:
                    logger.warning(f"[{execution_id}] No profile found or error fetching profile for user {event['user_id']}: {str(e)}")
                    notifications_failed += 1
                    continue
                if not user_profile or not user_profile.get('phone_number'):
                    logger.warning(f"[{execution_id}] No phone number found for user {event['user_id']} or profile incomplete.")
                    notifications_failed += 1
                    continue
                event_date_formatted = datetime.strptime(event['event_date'], '%Y-%m-%d').strftime('%B %d, %Y')
                event_type_text = "recurring event" if event['event_type'] == 'recurrence' else "deadline"
                user_name = user_profile.get('full_name', 'User')
                message = f"""
🔔 Event Reminder

Hi {user_name}!

You have an upcoming {event_type_text}:
📅 {event['title']}
📆 Date: {event_date_formatted}

Don't forget to prepare for this important event!

Best regards,
RemindMe
                """.strip()
                if send_whatsapp_notification(
                    phone_number=user_profile['phone_number'], 
                    message=message,
                    user_id=event['user_id'],
                    event_id=event_id,
                    notification_type='event_reminder'
                ):
                    notifications_sent += 1
                    logger.info(f"[{execution_id}] Notification successfully sent for event: {event['title']}")
                else:
                    notifications_failed += 1
                    logger.warning(f"[{execution_id}] Notification failed for event: {event['title']}.")
            finally:
                with processing_lock:
                    processing_events.discard(event_id)
                    logger.info(f"[{execution_id}] Removed event {event_id} from processing set.")
        logger.info(f"[{execution_id}] Notification check completed. Sent {notifications_sent} notifications, failed {notifications_failed}.")
    except Exception as e:
        logger.error(f"[{execution_id}] Critical error in check_and_send_notifications: {str(e)}")

# --- SCHEDULER FOR RSVP POLLING ---
def process_rsvp_messages():
    try:
        response = supabase.table('rsvp').select('*').or_('status.is.null,status.eq.""').execute()
        rsvps = response.data
        if not rsvps:
            logger.info("No pending RSVPs.")
            return
        for rsvp in rsvps:
            rsvp_id = rsvp['id']
            title = rsvp.get('title', 'Event')
            message_body = rsvp.get('message', '')
            logger.info(f"Processing RSVP id: {rsvp_id}")
            contacts_resp = supabase.table('rsvp_contact_status') \
                .select('*') \
                .eq('rsvp_id', rsvp_id) \
                .or_('invite_status.is.null,invite_status.eq.""') \
                .execute()
            contacts = contacts_resp.data
            if not contacts:
                logger.info(f"No contacts to send for rsvp_id {rsvp_id}")
                continue
            all_sent = True
            for contact in contacts:
                contact_id = contact['id']
                phone = contact['contact_phone']
                logger.info(f"Sending RSVP WhatsApp to {phone} for contact_id {contact_id}")
                try:
                    supabase.table('rsvp_contact_status') \
                        .update({'invite_status': 'processing'}) \
                        .eq('id', contact_id).execute()
                    sms_body = f"{title}\n{message_body}\n\nIf attending, respond by replying 'rsvp: yes'. If not, reply 'rsvp: no'."
                    msg = twilio_client.messages.create(
                        body=sms_body,
                        from_=twilio_whatsapp_number,
                        to=f"whatsapp:{phone}" if not str(phone).startswith("whatsapp:") else phone
                    )
                    logger.info(f"Sent to {phone}, Twilio SID: {msg.sid}")
                    supabase.table('rsvp_contact_status') \
                        .update({'invite_status': 'sent'}).eq('id', contact_id).execute()
                except Exception as e:
                    logger.error(f"Failed to send to {phone}: {e}")
                    supabase.table('rsvp_contact_status') \
                        .update({'invite_status': ''}).eq('id', contact_id).execute()
                    all_sent = False
            if all_sent:
                supabase.table('rsvp').update({'status': 'sent'}).eq('id', rsvp_id).execute()
                logger.info(f"All contacts sent for rsvp_id {rsvp_id}, updated RSVP status to sent.")
            else:
                logger.info(f"Some contacts failed for rsvp_id {rsvp_id}, will retry next run.")
    except Exception as e:
        logger.error(f"Error in process_rsvp_messages: {e}")

# --- SCHEDULER FOR MESSAGES POLLING ---
def process_messages():
    try:
        # 1. Get all messages with status NULL or empty
        response = supabase.table('messages').select('*').or_('status.is.null,status.eq.""').execute()
        messages = response.data
        if not messages:
            logger.info("No pending messages.")
            return
        for message in messages:
            message_id = message['id']
            message_text = message.get('content', 'Reminder!')
            logger.info(f"Processing message_id: {message_id}")
            # 2. Get all message_contact records for this message_id with status NULL or empty
            contacts_resp = supabase.table('message_contact') \
                .select('*') \
                .eq('message_id', message_id) \
                .or_('status.is.null,status.eq.""') \
                .execute()
            contacts = contacts_resp.data
            if not contacts:
                logger.info(f"No contacts to send for message_id {message_id}")
                continue
            all_sent = True
            for contact in contacts:
                contact_id = contact['id']
                phone = contact['contact_phone']
                logger.info(f"Sending WhatsApp to {phone} for contact_id {contact_id}")
                try:
                    # Mark as 'processing' to avoid duplicate sends
                    supabase.table('message_contact') \
                        .update({'status': 'processing'}) \
                        .eq('id', contact_id).execute()
                    # Send WhatsApp message
                    msg = twilio_client.messages.create(
                        body=message_text,
                        from_=twilio_whatsapp_number,
                        to=f"whatsapp:{phone}" if not str(phone).startswith("whatsapp:") else phone
                    )
                    logger.info(f"Sent to {phone}, Twilio SID: {msg.sid}")
                    # Mark as sent
                    supabase.table('message_contact') \
                        .update({'status': 'sent'}).eq('id', contact_id).execute()
                except Exception as e:
                    logger.error(f"Failed to send to {phone}: {e}")
                    # Optionally revert status so it retries later
                    supabase.table('message_contact') \
                        .update({'status': ''}).eq('id', contact_id).execute()
                    all_sent = False
            # 3. If all contacts were successfully sent, mark message as sent
            if all_sent:
                supabase.table('messages').update({'status': 'sent'}).eq('id', message_id).execute()
                logger.info(f"All contacts sent for message_id {message_id}, updated message status to sent.")
            else:
                logger.info(f"Some contacts failed for message_id {message_id}, will retry next run.")
    except Exception as e:
        logger.error(f"Error in process_messages: {e}")

# --- AI INTEGRATION FUNCTIONS ---
def get_user_id_from_whatsapp_number(whatsapp_number):
    phone_number_clean = whatsapp_number.replace('whatsapp:', '')
    try:
        response = supabase.table('profiles').select('id').eq('phone_number', phone_number_clean).single().execute()
        if response.data:
            return response.data['id']
        return None
    except Exception as e:
        logger.error(f"Error fetching user ID for {whatsapp_number}: {str(e)}")
        return None

def get_user_events(user_id, month=None, year=None, upcoming_only=False):
    try:
        query = supabase.table('events').select('*').eq('user_id', user_id)
        if month and year:
            start_date = f"{year}-{month:02d}-01"
            if month == 12:
                end_date = f"{year + 1}-01-01"
            else:
                end_date = f"{year}-{month + 1:02d}-01"
            query = query.gte('event_date', start_date).lt('event_date', end_date)
        elif upcoming_only:
            today = datetime.now().date().isoformat()
            query = query.gte('event_date', today)
        response = query.execute()
        return response.data
    except Exception as e:
        logger.error(f"Error fetching events for user {user_id}: {str(e)}")
        return []

def analyze_event_query(message_text):
    message_lower = message_text.lower()
    month_names = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    for month_name, month_num in month_names.items():
        if month_name in message_lower:
            return f'specific_month_{month_num}'
    if any(word in message_lower for word in ['this month', 'current month', 'month']):
        return 'current_month'
    elif any(word in message_lower for word in ['next month', 'following month']):
        return 'next_month'
    elif any(word in message_lower for word in ['this week', 'current week', 'week']):
        return 'current_week'
    elif any(word in message_lower for word in ['upcoming', 'future', 'coming', 'ahead']):
        return 'upcoming'
    elif any(word in message_lower for word in ['all', 'everything', 'list']):
        return 'all'
    else:
        return 'current_month'

def get_gemini_response(user_id, message_text):
    if not gemini_model:
        return "I'm sorry, the AI service is currently unavailable. Please try again later."
    with conversations_lock:
        current_time = datetime.now()
        if user_id not in conversations:
            conversations[user_id] = {
                'last_message_time': current_time,
                'history': []
            }
            logger.info(f"Started new conversation for user {user_id}.")
        conversations[user_id]['last_message_time'] = current_time
        try:
            all_user_events = get_user_events(user_id)
            current_date = datetime.now()
            events_context = ""
            if all_user_events:
                events_context = f"\n\nHere are ALL your events from the database:\n"
                for event in all_user_events:
                    event_date = datetime.strptime(event['event_date'], '%Y-%m-%d').strftime('%B %d, %Y')
                    event_type = "recurring event" if event['event_type'] == 'recurrence' else "deadline"
                    days_until = (datetime.strptime(event['event_date'], '%Y-%m-%d').date() - current_date.date()).days
                    if days_until == 0:
                        time_info = " (today)"
                    elif days_until == 1:
                        time_info = " (tomorrow)"
                    elif days_until > 0:
                        time_info = f" (in {days_until} days)"
                    else:
                        time_info = f" ({abs(days_until)} days ago)"
                    events_context += f"- {event['title']} ({event_type}) on {event_date}{time_info}\n"
                events_context += f"\nTotal: {len(all_user_events)} event(s) in your database."
            else:
                events_context = f"\n\nYou have no events in your database."
            message_lower = message_text.lower()
            if any(keyword in message_lower for keyword in ['event', 'events', 'schedule', 'calendar', 'reminder', 'deadline', 'appointment', 'meeting', 'september', 'month', 'week']):
                query_type = analyze_event_query(message_text)
                if query_type.startswith('specific_month_'):
                    month_num = int(query_type.split('_')[2])
                    target_year = current_date.year
                    if month_num < current_date.month:
                        target_year = current_date.year + 1
                    filtered_events = get_user_events(user_id, month_num, target_year)
                    period_text = f"{datetime(target_year, month_num, 1).strftime('%B %Y')}"
                elif query_type == 'current_month':
                    current_month_events = get_user_events(user_id, current_date.month, current_date.year)
                    period_text = f"{current_date.strftime('%B %Y')}"
                    filtered_events = current_month_events
                elif query_type == 'next_month':
                    next_month = current_date.month + 1 if current_date.month < 12 else 1
                    next_year = current_date.year if current_date.month < 12 else current_date.year + 1
                    filtered_events = get_user_events(user_id, next_month, next_year)
                    period_text = f"{datetime(current_date.year, next_month, 1).strftime('%B %Y')}"
                elif query_type == 'current_week':
                    upcoming_events = get_user_events(user_id, upcoming_only=True)
                    week_start = current_date - timedelta(days=current_date.weekday())
                    week_end = week_start + timedelta(days=7)
                    filtered_events = [e for e in upcoming_events if week_start.date() <= datetime.strptime(e['event_date'], '%Y-%m-%d').date() < week_end.date()]
                    period_text = f"this week (starting {week_start.strftime('%B %d')})"
                elif query_type == 'upcoming':
                    filtered_events = get_user_events(user_id, upcoming_only=True)
                    period_text = "upcoming"
                else:
                    filtered_events = all_user_events
                    period_text = "all time"
                if filtered_events:
                    events_context += f"\n\nFiltered events for {period_text}:\n"
                    for event in filtered_events:
                        event_date = datetime.strptime(event['event_date'], '%Y-%m-%d').strftime('%B %d, %Y')
                        event_type = "recurring event" if event['event_type'] == 'recurrence' else "deadline"
                        days_until = (datetime.strptime(event['event_date'], '%Y-%m-%d').date() - current_date.date()).days
                        if days_until == 0:
                            time_info = " (today)"
                        elif days_until == 1:
                            time_info = " (tomorrow)"
                        elif days_until > 0:
                            time_info = f" (in {days_until} days)"
                        else:
                            time_info = f" ({abs(days_until)} days ago)"
                        events_context += f"- {event['title']} ({event_type}) on {event_date}{time_info}\n"
                    events_context += f"\nFiltered total: {len(filtered_events)} event(s) for {period_text}"
                else:
                    events_context += f"\n\nNo events found for {period_text}."
            enhanced_message = f"{message_text}\n\nContext: You are an AI assistant for RemindMe, a reminder app. You have access to the user's event database. Use this information to answer their questions about events, schedules, reminders, etc.{events_context}"
            chat = gemini_model.start_chat(history=conversations[user_id]['history'])
            response = chat.send_message(enhanced_message)
            conversations[user_id]['history'].append({'role': 'user', 'parts': [message_text]})
            conversations[user_id]['history'].append({'role': 'model', 'parts': [response.text]})
            logger.info(f"Gemini response for user {user_id}: {response.text}")
            return response.text
        except Exception as e:
            logger.error(f"Error getting Gemini response for user {user_id}: {str(e)}")
            return "I'm sorry, I couldn't process your request at the moment. Please try again."

# --- UNIFIED TWILIO WEBHOOK ROUTE ---
@app.route('/webhook/twilio', methods=['POST'])
def twilio_webhook():
    incoming_msg = request.values.get('Body', '').strip()
    from_whatsapp_number = request.values.get('From', '').strip()
    logger.info(f"Received message from {from_whatsapp_number}: '{incoming_msg}'")
    # RSVP logic
    if incoming_msg.lower().startswith('rsvp:'):
        return handle_rsvp_reply(from_whatsapp_number, incoming_msg)
    # AI logic (Gemini, event queries, etc.)
    else:
        return handle_ai_conversation(from_whatsapp_number, incoming_msg)

# --- RSVP REPLY HANDLER ---
def handle_rsvp_reply(from_number, body):
    resp = MessagingResponse()
    # Normalize phone number (remove 'whatsapp:' prefix if present)
    normalized_number = from_number.replace('whatsapp:', '') if from_number and from_number.startswith('whatsapp:') else from_number
    match = re.search(r"rsvp:\s*(yes|no)", body.lower())
    if not match:
        resp.message("Sorry, I didn't understand your RSVP response. Please reply with 'rsvp: yes' or 'rsvp: no'.")
        return Response(str(resp), mimetype='text/xml')
    response = match.group(1)
    status_value = 'yes' if response == 'yes' else 'no'
    try:
        contact_resp = supabase.table('rsvp_contact_status') \
            .select('*') \
            .eq('contact_phone', normalized_number) \
            .eq('invite_status', 'sent') \
            .order('id', desc=True) \
            .limit(1) \
            .execute()
        contacts = contact_resp.data
        if contacts:
            contact_id = contacts[0]['id']
            supabase.table('rsvp_contact_status') \
                .update({'status': status_value}) \
                .eq('id', contact_id).execute()
            logger.info(f"Updated RSVP status for contact_id {contact_id} to {status_value}")
            # Send confirmation WhatsApp message back to the user
            try:
                confirmation_msg = "Thanks for your response! Your RSVP has been recorded."
                if from_number:
                    twilio_client.messages.create(
                        body=confirmation_msg,
                        from_=twilio_whatsapp_number,
                        to=from_number
                    )
                    logger.info(f"Sent confirmation to {from_number}")
            except Exception as e:
                logger.error(f"Failed to send confirmation to {from_number}: {e}")
            resp.message(confirmation_msg)
        else:
            resp.message("Sorry, we could not find your RSVP invitation. Please ensure you are replying from the same WhatsApp number you received the invite on.")
    except Exception as e:
        logger.error(f"Error updating RSVP status: {e}")
        resp.message("Sorry, there was an error recording your RSVP. Please try again later.")
    return Response(str(resp), mimetype='text/xml')

# --- AI CONVERSATION HANDLER ---
def handle_ai_conversation(from_whatsapp_number, incoming_msg):
    resp = MessagingResponse()
    user_id = get_user_id_from_whatsapp_number(from_whatsapp_number)
    if not user_id:
        logger.warning(f"Could not find user_id for WhatsApp number: {from_whatsapp_number}. Cannot start AI conversation.")
        resp.message("I'm sorry, I can't identify your account. Please ensure your phone number is registered in our system.")
        return Response(str(resp), mimetype='text/xml')
    gemini_reply = get_gemini_response(user_id, incoming_msg)
    phone_number_clean = from_whatsapp_number.replace('whatsapp:', '')
    log_notification_to_db(
        user_id=user_id,
        event_id=None,
        notification_type='ai_response',
        notification_content=gemini_reply,
        phone_number=phone_number_clean,
        twilio_message_sid=None,
        delivery_status='sent'
    )
    resp.message(gemini_reply)
    return Response(str(resp), mimetype='text/xml')

# --- SCHEDULER THREAD ---
def run_scheduler():
    schedule.every().day.at("00:43").do(check_and_send_notifications)
    # schedule.every().minute.do(check_and_send_notifications)  # For testing
    logger.info("Scheduler started. Notifications will be sent daily at 21:20")
    while True:
        schedule.run_pending()
        time.sleep(60)

# --- APSCHEDULER FOR RSVP POLLING ---
scheduler = BackgroundScheduler()
scheduler.add_job(func=process_rsvp_messages, trigger="interval", seconds=15)
scheduler.add_job(func=process_messages, trigger="interval", seconds=15)
scheduler.start()

# --- FLASK ROUTES ---
@app.route('/check-notifications', methods=['GET'])
def manual_check_notifications():
    try:
        check_and_send_notifications()
        return jsonify({
            'status': 'success',
            'message': 'Manual notification check completed. Check server logs for details.'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/test-notification/<user_id>', methods=['GET'])
def test_notification(user_id):
    try:
        user_response = supabase.table('profiles').select('full_name, phone_number').eq('id', user_id).single().execute()
        user_profile = user_response.data
        if not user_profile or not user_profile.get('phone_number'):
            return jsonify({
                'status': 'error',
                'message': 'User not found or no phone number'
            }), 404
        test_message = f"""
🔔 Test Notification

Hi {user_profile.get('full_name', 'User')}!

This is a test message from RemindMe to verify WhatsApp notifications are working correctly.

If you received this message, your notifications are set up properly! 🎉
        """.strip()
        if send_whatsapp_notification(
            phone_number=user_profile['phone_number'], 
            message=test_message,
            user_id=user_id,
            event_id=None,
            notification_type='test_notification'
        ):
            return jsonify({
                'status': 'success',
                'message': f'Test notification sent to {user_profile["phone_number"]}'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to send test notification'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'RemindMe Notification API with Scheduler and AI'
    })

@app.route('/notifications/<user_id>', methods=['GET'])
def get_user_notifications(user_id):
    try:
        response = supabase.table('notifications_sent').select(
            'id, event_id, notification_type, notification_content, phone_number, delivery_status, sent_at'
        ).eq('user_id', user_id).order('sent_at', desc=True).limit(50).execute()
        return jsonify({
            'status': 'success',
            'notifications': response.data
        })
    except Exception as e:
        logger.error(f"Error fetching notifications for user {user_id}: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# --- SCHEDULER THREAD START ---
_scheduler_started = False
if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        if not _scheduler_started:
            scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
            scheduler_thread.start()
            _scheduler_started = True
            logger.info("Scheduler thread started in main process.")
        else:
            logger.info("Scheduler thread already started.")
    else:
        logger.info("Skipping scheduler thread start in reloader process.")
    logger.info("Starting RemindMe Notification API with built-in scheduler and AI...")
    app.run(debug=True, host='0.0.0.0', port=5002)
