import os
from dotenv import load_dotenv
from langchain_google_community.calendar.create_event import CalendarCreateEvent
from langchain_googledrive.tools.google_drive import GoogleDriveCreateFileTool
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
        drive_tool = GoogleDriveCreateFileTool(
            api_wrapper=GoogleDriveAPIWrapper(folder_id="root", num_results=2)
        )
        result = drive_tool.run({
            "name": filename,
            "data": summary_text
        })
        return f"Summary saved to Google Drive: {result}"
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
        calendar_tool = CalendarCreateEvent(
            calendar_id='primary',
            summary=summary_text,
            start_time=datetime.datetime.now(),
            end_time=datetime.datetime.now() + datetime.timedelta(hours=1)
        )
        result = calendar_tool.run(summary_text)
        return f"Calendar event created: {result}"
    except Exception as e:
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
