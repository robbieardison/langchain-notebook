from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain.agents import initialize_agent
from pydantic import BaseModel, Field
from typing import Type
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from dotenv import load_dotenv
import os
import datetime
import re

# ------------- ENV SETUP ------------- #
load_dotenv()
client_secret_path = os.getenv("GOOGLE_CLIENT_SECRET_PATH")

# ------------- LLM SETUP ------------- #
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

summary_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="Summarize this meeting transcript:\n{transcript}\nProvide the summary and bullet point action items."
)

calendar_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="Extract any date/time for follow-up meetings or scheduled events:\n{transcript}"
)

summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
calendar_chain = LLMChain(llm=llm, prompt=calendar_prompt)

# ------------- GOOGLE API SETUP ------------- #
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

# ------------- INPUT SCHEMAS ------------- #
class TextInput(BaseModel):
    text: str = Field(description="Plain text input")

# ------------- TOOL DEFINITIONS ------------- #
class SummarizationTool(BaseTool):
    name: str = "Summarization"
    description: str = "Summarizes a meeting transcript."
    args_schema: Type[BaseModel] = TextInput

    def _run(self, text: str):
        return summary_chain.invoke({"transcript": text})['text']
    
    def _arun(self, text: str):
        raise NotImplementedError

class GoogleDriveTool(BaseTool):
    name: str = "Google Drive"
    description: str = "Uploads a text file to Google Drive."
    args_schema: Type[BaseModel] = TextInput

    def _run(self, text: str):
        file_name = "Meeting Summary.txt"
        with open(file_name, 'w') as f:
            f.write(text)

        file_metadata = {
            'name': file_name,
            'mimeType': 'text/plain'
        }

        media = MediaFileUpload(file_name, mimetype='text/plain')
        file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return f"File uploaded to Drive with ID: {file['id']}"

    def _arun(self, text: str):
        raise NotImplementedError

class GoogleCalendarTool(BaseTool):
    name: str = "Google Calendar"
    description: str = "Creates a calendar event from structured meeting info."
    args_schema: Type[BaseModel] = TextInput

    def _run(self, text: str):
        match = re.search(r"Title: (.*?)\nDate: (.*?)\nTime: (.*?)\nDescription: (.*?)$", text, re.DOTALL)
        if not match:
            return "No event info found."

        title, date_str, time_str, description = match.groups()
        
        # Default to the next week if no date is found
        if not date_str:
            date_str = (datetime.datetime.now() + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        
        # Default to 10 AM if no time is found
        if not time_str:
            time_str = "10:00 AM"
        
        try:
            start_time = self.parse_event_datetime(date_str, time_str)
            end_time = start_time + datetime.timedelta(hours=1)

            event = {
                'summary': title,
                'description': description,
                'start': {'dateTime': start_time.isoformat(), 'timeZone': 'Asia/Jakarta'},
                'end': {'dateTime': end_time.isoformat(), 'timeZone': 'Asia/Jakarta'}
            }

            created_event = calendar_service.events().insert(calendarId='primary', body=event).execute()
            return f"Event created: {created_event.get('htmlLink')}"
        except Exception as e:
            return f"Error parsing event: {str(e)}"


# ------------- AGENT SETUP ------------- #
tools = [SummarizationTool(), GoogleDriveTool(), GoogleCalendarTool()]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=True
)

# ------------- MAIN PROCESS ------------- #
def process_meeting(transcript):
    summary = agent.run(f"Summarize this meeting transcript:\n{transcript}")
    calendar_info = agent.run(
        f"""Check this meeting transcript and extract any date and time for follow-up meetings or scheduled events.
            Format the output like this if relevant:

            Title: <title of the meeting>
            Date: <YYYY-MM-DD>
            Time: <HH:MM>
            Description: <brief reason for the event>

            Transcript:
            {transcript}

            If someone says 'let's follow up next week' and no date is given, default to the week after today at 10am.:\n{transcript}""")
    event_link = agent.run(f"Create a Google Calendar event from the following info:\n{calendar_info}")
    drive_upload = agent.run(f"Upload the meeting summary to Google Drive:\n{summary}")
    
    print(summary)
    print(event_link)
    print(drive_upload)

# Example usage
if __name__ == "__main__":
    with open("sample_meeting.txt", "r") as f:
        transcript = f.read()
    process_meeting(transcript)
