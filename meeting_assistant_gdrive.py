from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from dotenv import load_dotenv
import os
import datetime
import re


load_dotenv()
client_secret_path = os.getenv("GOOGLE_CLIENT_SECRET_PATH")

# ------------- LLM SETUP ------------- #
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

summary_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="""
    Summarize this meeting transcript:

    {transcript}
    
    Provide the summary and bullet point action items.
    """
)

calendar_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="""
    Check this meeting transcript and extract any date and time for follow-up meetings or scheduled events.
    Format the output like this if relevant:

    Title: <title of the meeting>
    Date: <YYYY-MM-DD>
    Time: <HH:MM>
    Description: <brief reason for the event>

    Transcript:
    {transcript}

    If someone says 'let's follow up next week' and no date is given, default to the week after today at 10am."
    """
)

summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
calendar_chain = LLMChain(llm=llm, prompt=calendar_prompt)

# ------------- GOOGLE AUTH ------------- #
SCOPES = [
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/calendar'
]

creds = None
if os.path.exists('token.json'):
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
        creds = flow.run_local_server(port=0)
    with open('token.json', 'w') as token:
        token.write(creds.to_json())

calendar_service = build('calendar', 'v3', credentials=creds)
drive_service = build('drive', 'v3', credentials=creds)

# ------------- CORE FUNCTIONS ------------- #
def analyze_transcript(transcript):
    summary = summary_chain.invoke({"transcript": transcript})
    calendar_info = calendar_chain.invoke({"transcript": transcript})
    return summary['text'], calendar_info['text']

def create_calendar_event(event_text):
    match = re.search(r"Title: (.*?)\nDate: (.*?)\nTime: (.*?)\nDescription: (.*?)$", event_text, re.DOTALL)
    if match:
        print("Matched Event:")
        for i, group in enumerate(match.groups()):
            print(f"Group {i+1}: {group}")
    else:
        print("No match found.")
        return None

    title, date_str, time_str, description = match.groups()
    def parse_event_datetime(date_str, time_str):
        date_str = date_str.strip()
        time_str = re.sub(r"\s+", " ", time_str.strip())
        datetime_formats = [
            "%Y-%m-%d %I:%M %p",  # e.g., 2025-04-20 10:00 AM
            "%Y-%m-%d %H:%M"      # e.g., 2025-04-20 14:00
        ]
        
        for fmt in datetime_formats:
            try:
                return datetime.datetime.strptime(f"{date_str} {time_str}", fmt)
            except ValueError:
                continue
        raise ValueError(f"Unrecognized date/time format: {date_str} {time_str}")
    
    event_date = parse_event_datetime(date_str, time_str)
    end_time = event_date + datetime.timedelta(hours=1)

    event = {
        'summary': title,
        'description': description,
        'start': {'dateTime': event_date.isoformat(), 'timeZone': 'Asia/Jakarta'},
        'end': {'dateTime': end_time.isoformat(), 'timeZone': 'Asia/Jakarta'}
    }

    created_event = calendar_service.events().insert(calendarId='primary', body=event).execute()
    return created_event.get('htmlLink')

def upload_to_drive(file_name: str, content: str):
    # Save content to a temporary file
    with open(file_name, 'w') as f:
        f.write(content)
    
    # File metadata
    file_metadata = {
        'name': file_name,
        'mimeType': 'text/plain'
    }
    
    # Create a MediaFileUpload object for the file
    media = MediaFileUpload(file_name, mimetype='text/plain')
    
    # Upload the file to Google Drive
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f'File uploaded successfully. File ID: {file["id"]}')
    return file["id"]

# ------------- MAIN ------------- #
if __name__ == "__main__":
    with open("sample_meeting.txt", "r") as f:
        transcript = f.read()

    summary_text, calendar_text = analyze_transcript(transcript)

    print("\n--- SUMMARY ---\n")
    print(summary_text)
    print("\n--- CALENDAR EXTRACTION ---\n")
    print(calendar_text)

    event_link = create_calendar_event(calendar_text)
    if event_link:
        print("\nEvent created:", event_link)
    else:
        print("\nNo event found in transcript.")

    doc_id = upload_to_drive("Meeting Summary.txt", summary_text)
    print(f"\nSummary uploaded to Google Drive with ID: {doc_id}")