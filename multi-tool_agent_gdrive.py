from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Google Drive imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# --- Google Drive Upload Tool ---
SCOPES = [
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/calendar'
]

def upload_to_drive(file_path, file_name, mime_type='application/vnd.google-apps.document'):
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('drive', 'v3', credentials=creds)
    file_metadata = {
        'name': file_name,
        'mimeType': mime_type
    }
    media = MediaFileUpload(file_path, mimetype='text/plain')

    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()
    return f"File uploaded to Google Drive with ID: {file.get('id')}"

@tool
def summarize_meeting(transcript: str) -> str:
    """Summarize the meeting transcript."""
    return llm.invoke(f"Summarize this: {transcript}")

@tool
def extract_tasks(transcript: str) -> str:
    """Extract action items from the meeting transcript."""
    return llm.invoke(f"Extract tasks: {transcript}")

# LangChain Tool for uploading
@tool
def save_summary_to_drive(summary: str) -> str:
    """Saves the meeting summary to Google Drive."""
    with open("meeting_summary.txt", "w") as f:
        f.write(summary)
    return upload_to_drive("meeting_summary.txt", "Meeting Summary")

# LangChain setup
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

agent = initialize_agent(
    tools=[save_summary_to_drive],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=True
)

# Run agent
response = agent.invoke("This is the summary of our meeting. Please save it to Google Drive.")
print(response)
