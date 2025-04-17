import os
import re
from dotenv import load_dotenv
import datetime
from langchain_google_community.calendar.create_event import CalendarCreateEvent
from langchain_google_community.calendar.utils import get_google_credentials
from langchain_googledrive.utilities.google_drive import GoogleDriveAPIWrapper
from langchain.agents import initialize_agent, AgentType, tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI


# Ensure the environment variable for Google credentials is set
load_dotenv()
client_secret_path = os.getenv("GOOGLE_CLIENT_SECRET_PATH")

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Global variable to store the summary
summary_text = ""

# Tool to summarize the meeting transcript
@tool
def summarize_meeting(file_path: str) -> str:
    """Summarizes the meeting transcript from the specified file."""
    global summary_text
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        prompt = f"Summarize the following meeting transcript:\n\n{content}"
        response = llm.predict(prompt)
        summary_text = response
        print(summary_text)  # Debugging
        return "Meeting summarized successfully."
    except Exception as e:
        return f"Error summarizing meeting: {e}"

# Tool to save the summary to Google Drive
@tool
def save_summary_to_drive(filename: str) -> str:
    """Saves the summary to Google Drive with the given filename."""
    global summary_text
    if not summary_text:
        return "No summary available. Please summarize the meeting first."
    try:
        drive_api = GoogleDriveAPIWrapper(scopes=['https://www.googleapis.com/auth/drive.file'])
        # Create a file in Google Drive
        file_metadata = {
            'name': 'meeting_summary_with_drive_api.txt',
            'mimeType': 'text/plain'
        }
        file_content = summary_text
        file_id = drive_api.create_file(file_metadata, file_content)
        return f"Summary saved to Google Drive with id: {file_id}"
    except Exception as e:
        return f"Error saving to Google Drive: {e}"

# Tool to create a calendar event based on the summary
@tool
def create_calendar_event_from_summary() -> str:
    """Creates a calendar event based on the summarized meeting."""
    global summary_text
    if not summary_text:
        return "No summary available. Please summarize the meeting first."
    try:
        # Parse the summary_text to find mentions of new meetings or events
        match_specific = re.search(
            r"\*\*Next Meeting:\*\*.*?(?P<date>\w+ \d{1,2}, \d{4}).*?(?P<time>\d{1,2}:\d{2} (?:AM|PM))",
            summary_text,
            re.IGNORECASE
        )
        if match_specific:
            # Extract specific date and time
            date_str = match_specific.group("date")
            time_str = match_specific.group("time")
            event_datetime = datetime.datetime.strptime(f"{date_str} {time_str}", "%B %d, %Y %I:%M %p")
        else:
            # Check for "following week" phrasing
            match_following_week = re.search(r"\*\*Next Meeting", summary_text, re.IGNORECASE)
            if not match_following_week:
                return "No mention of a new meeting or event found in the summary."

            # Infer the date and time for the next meeting
            today = datetime.date.today()
            next_week = today + datetime.timedelta(weeks=1)
            event_datetime = datetime.datetime.combine(next_week, datetime.time(10, 0))  # Default to 10:00 AM

        # Set default duration (1 hour)
        end_datetime = event_datetime + datetime.timedelta(hours=1)

        # Format datetime as "YYYY-MM-DD HH:MM:SS"
        start_datetime_formatted = event_datetime.strftime("%Y-%m-%d %H:%M:%S")
        end_datetime_formatted = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Prepare the payload for the calendar event
        payload = {
            "calendar_id": "primary",
            "summary": "Follow-up Meeting",
            "description": summary_text,
            "start_datetime": start_datetime_formatted,
            "end_datetime": end_datetime_formatted,
            "timezone": "Asia/Jakarta"
        }
        print("Payload for Calendar Event:", payload)  # Debugging

        # Create the calendar event
        calendar_tool = CalendarCreateEvent(
            scopes=['https://www.googleapis.com/auth/calendar'],
            credentials_path=client_secret_path  # Use the client_secret_path
        )
        result = calendar_tool.run(payload)
        print("API Response:", result)  # Debugging
        return f"Calendar event created: {result}"
    except Exception as e:
        print(f"Error details: {e}")  # Debugging
        return f"Error creating calendar event: {e}"

# Initialize the agent with the defined tools
tools = [summarize_meeting, save_summary_to_drive, create_calendar_event_from_summary]
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    memory=memory
)

# Command-line interface loop
def main():
    print("Welcome to the Meeting Assistant!")
    print("Available commands:")
    print("- Summarize the meeting from sample_meeting.txt")
    print("- Save summary to drive with filename 'Meeting_Summary.txt'")
    print("- Create calendar event from summary")
    print("- Exit")

    while True:
        user_input = input("\nEnter your command: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = agent.run(user_input)
        print(f"\n{response}")

if __name__ == "__main__":
    main()
