from pydantic import BaseModel, Field
# from langchain.tools import PythonAstREPLTool
from langchain_experimental.tools import PythonAstREPLTool
import os
import streamlit as st
from airtable import Airtable
from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from pytz import timezone
import datetime
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI
import langchain
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.smith import RunEvalConfig, run_on_dataset
import pandas as pd

hide_share_button_style = """
    <style>
    .st-emotion-cache-zq5wmm.ezrtsby0 .stActionButton:nth-child(1) {
        display: none !important;
    }
    </style>
"""

hide_star_and_github_style = """
    <style>
    .st-emotion-cache-1lb4qcp.e3g6aar0,
    .st-emotion-cache-30do4w.e3g6aar0 {
        display: none !important;
    }
    </style>
"""

hide_mainmenu_style = """
    <style>
    #MainMenu {
        display: none !important;
    }
    </style>
"""

hide_fork_app_button_style = """
    <style>
    .st-emotion-cache-alurl0.e3g6aar0 {
        display: none !important;
    }
    </style>
"""

st.markdown(hide_share_button_style, unsafe_allow_html=True)
st.markdown(hide_star_and_github_style, unsafe_allow_html=True)
st.markdown(hide_mainmenu_style, unsafe_allow_html=True)
st.markdown(hide_fork_app_button_style, unsafe_allow_html=True)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
st.image("Twitter.jpg")

datetime.datetime.now()
current_date = datetime.date.today().strftime("%m/%d/%y")
day_of_week = datetime.date.today().weekday()
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
current_day = days[day_of_week]
todays_date = current_date
day_of_the_week = current_day

business_details_text = [
    "working days: all Days except sunday",
    "working hours: 9 am to 7 pm", 
    "Phone: (555) 123-4567",
    "Address: 567 Oak Avenue, Anytown, CA 98765, Email: jessica.smith@example.com",
    "dealer ship location: https://www.google.com/maps/place/Pine+Belt+Mazda/@40.0835762,-74.1764688,15.63z/data=!4m6!3m5!1s0x89c18327cdc07665:0x23c38c7d1f0c2940!8m2!3d40.0835242!4d-74.1742558!16s%2Fg%2F11hkd1hhhb?entry=ttu"
]
retriever_3 = FAISS.from_texts(business_details_text, OpenAIEmbeddings()).as_retriever()

file_1 = r'dealer_1_inventry.csv'

loader = CSVLoader(file_path=file_1)
docs_1 = loader.load()
embeddings = OpenAIEmbeddings()
vectorstore_1 = FAISS.from_documents(docs_1, embeddings)
retriever_1 = vectorstore_1.as_retriever(search_type="similarity", search_kwargs={"k": 3})#check without similarity search and k=8


# Create the first tool
# tool1 = create_retriever_tool(
#     retriever_1, 
#      "search_car_dealership_inventory",
#      "This tool is used when answering questions related to car inventory.\
#       Searches and returns documents regarding the car inventory. Input to this can be multi string.\
#       The primary input for this function consists of either the car's make and model, whether it's new or used car, and trade-in.\
#       You should know the make of the car, the model of the car, and whether the customer is looking for a new or used car to answer inventory-related queries.\
#       When responding to inquiries about any car, restrict the information shared with the customer to the car's make, year, model, and trim.\
#       The selling price should only be disclosed upon the customer's request, without any prior provision of MRP.\
#       If the customer inquires about a car that is not available, please refrain from suggesting other cars.\
#       Provide a link for more details after every car information given."
# )
tool1 = create_retriever_tool(
    retriever_1, 
     "search_car_model_make",
     "This tool is used only when you know model of the car or features of the car for example good mileage car, toeing car,\
     pickup truck or and new or used car and \
      Searches and returns documents regarding the car details. Input to this should be the car's model or car features and new or used car as a single argument"
) 


# Create the third tool
tool3 = create_retriever_tool(
    retriever_3, 
    "search_business_details",
    "Searches and returns documents related to business working days and hours, location and address details."
)

# Append all tools to the tools list
airtable_api_key = st.secrets["AIRTABLE"]["AIRTABLE_API_KEY"]
os.environ["AIRTABLE_API_KEY"] = airtable_api_key
AIRTABLE_BASE_ID = "appN324U6FsVFVmx2"  
AIRTABLE_TABLE_NAME = "car_details_input"

# Streamlit UI setup
st.info("Introducing **Otto**, your cutting-edge partner in streamlining dealership and customer-related operations. At EngagedAi, we specialize in harnessing the power of automation to revolutionize the way dealerships and customers interact. Our advanced solutions seamlessly handle tasks, from managing inventory and customer inquiries to optimizing sales processes, all while enhancing customer satisfaction. Discover a new era of efficiency and convenience with us as your trusted automation ally. [engagedai.io](https://funnelai.com/). For this demo application, we will use the Inventory Dataset. Please explore it [here](https://github.com/ShahVishs/workflow/blob/main/2013_Inventory.csv) to get a sense for what questions you can ask.")
# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'generated' not in st.session_state:
    st.session_state.generated = []
if 'past' not in st.session_state:
    st.session_state.past = []
# Initialize user name in session state
if 'user_name' not in st.session_state:
    st.session_state.user_name = None

llm = ChatOpenAI(model="gpt-4", temperature = 0)
langchain.debug=True
# memory_key = "history"
# memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)
memory_key="chat_history"
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
template = """You are an costumer care support at car dealership responsible for handling inquiries related to 
car inventory, business details and appointment scheduling.
To ensure a consistent and effective response, please adhere to the following guidelines:

Car Inventory Inquiries:
In our dealership, we offer a wide selection of vehicles from various manufacturers, Now Our dealership has only 
{available_makers} maker cars each designed to meet specific needs, including towing, off-road capabilities, good mileage,
pickup trucks.
If a customer inquires about our car inventory with features related to towing, off-road capability, good mileage, or pickup 
trucks, there's no need to ask about the make and model of the car. Simply inquire whether they are interested in a new or
used vehicle.


Car Variety:
Recognize that the dealership offers a wide variety of car makes.
Understand that each make may have multiple models available in the inventory without knowing exact 
model you should not give details. 
For example "Jeep is a make and Jeep Cherokee, Jeep Wrangler, Jeep Grand Cherokee are models
similarly Ram is a maker and Ram 1500, Ram 2500 and Ram 3500 are models"
Please note that the above provided make and model details of jeep and ram in double inverted coomas are solely for 
illustration purposes and should not be used to respond customer queries.

Identify Query Content:
When customers make inquiries, carefully examine the content of their question.
Determine whether their inquiry contains information about the car's make, model, or both.

Model Identification:
To assist customers effectively, identify the specific model of the car they are interested in.

Request Missing Model:
If the customer's inquiry mentions only the car's make (manufacturer):
Proactively ask them to provide the model information.
This step is crucial because multiple models can be associated with a single make.

New or Used Car Preference:
After identifying the car model, inquire about the customer's preference for a new or used car.
Understanding their preference will help tailor the recommendations to their specific needs.

Ask only one question at a time like when asking about model dont ask used or new car. First ask model than 
used or new car separatly.
You should give details of the available cars in inventory only when you get the above details. i.e model, new or used car.

Part 2:
In Part 1 You gather Make, Model, and New/Used info from the customer.
strictly follow If you have model and new or used car information from the user than only 
proceed to provide car details Make, Year, Model, Trim, separatly along with links for more information without square brackets.
Selling Price Disclosure:
Disclose the selling price of a car only when the customer explicitly requests it.
Do not provide the price in advance or mention the Maximum Retail Price (MRP).
One crucial piece of information to note is that you will be provided with information for a maximum of three cars from 
the inventory file. However, it's possible that there are more than three cars that match the customer's interest. 
In such cases, your response should be framed to convey that we have several models available. 
Here's a suggested response format:
"We have several models available. Here are a few options:"
If the customer's query matches a car model, respond with a list of car without square brackets, 
including the make, year, model, and trim, and provide their respective links in the answer.

Checking Appointments Avaliability: If the customer's inquiry lacks specific details such as their preferred/
day, date or time kindly engage by asking for these specifics.
{details} Use these details that is todays date and day and find the appointment date from the users input
and check for appointment availabity using function mentioned in the tools for 
that specific day or date and time.
For checking appointment vailability you use pandas dataframe in Python. The name of the dataframe is `df`. The dataframe contains 
data related appointment schedule. It is important to understand the attributes of the dataframe before working with it. 
This is the result of running `df.head().to_markdown()`. Important rule is set the option to display all columns without
truncation while using pandas.
<df>
{dhead}
</df>
You are not meant to use only these rows to answer questions - they are meant as a way of telling you
about the shape and schema of the dataframe.
you can run intermediate queries to do exporatory data analysis to give you more information as needed.

If the appointment slot for the requested date and time is not available, we can offer alternative times that are close to the customer's preferred time based 
on the information provided.

Additionally, use  Markdown format '[click here](www.12345.com).' to create a clickable link and  When they click on this link, it will take them to a URL
where they can schedule their appointment themselves."

Prior to scheduling an appointment, please commence a conversation by soliciting the following customer information:
first ask If they have a car for trade-in then separatly ask for their name, contact number and email address.
Business details: Enquiry regarding google maps location of the store, address of the store, working days and working hours 
and contact details use search_business_details tool to get information.

Encourage Dealership Visit: Our goal is to encourage customers to visit the dealership for test drives or
receive product briefings from our team. After providing essential information on the car's make, model,
color, and basic features, kindly invite the customer to schedule an appointment for a test drive or visit us
for a comprehensive product overview by our experts.

Make every effort to assist the customer promptly.
Keep responses concise, not exceeding two sentences.
"""

details= "Today's current date is "+ todays_date +" today's weekday is "+day_of_the_week+"."
available_makers="Chrysler, Jeep, Ram"

class PythonInputs(BaseModel):
    query: str = Field(description="code snippet to run")

df = pd.read_csv("appointment_new.csv")
# input_template = template.format(dhead=df.head().to_markdown(),details=details)
input_template = template.format(dhead=df.head().to_markdown(),details=details,available_makers=available_makers)
system_message = SystemMessage(content=input_template)

prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
)

repl = PythonAstREPLTool(locals={"df": df}, name="python_repl",
    description="Use to check on available appointment times for a given date and time. The input to this tool should be a string in this format mm/dd/yy. This is the only way for you to answer questions about available appointments. This tool will reply with available times for the specified date in 24hour time, for example: 15:00 and 3pm are the same")

tools = [tool1, repl, tool3]
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
if 'agent_executor' not in st.session_state:
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, return_source_documents=True,
        return_generated_question=True, return_intermediate_steps=True)
    st.session_state.agent_executor = agent_executor
else:
    agent_executor = st.session_state.agent_executor

chat_history=[]

response_container = st.container()
container = st.container()

airtable = Airtable(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME, api_key=airtable_api_key)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'user_name' not in st.session_state:
    st.session_state.user_name = None

# Function to save chat history to Airtable
def save_chat_to_airtable(user_name, user_input, output):
    try:
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        airtable.insert(
            {
                "username": user_name,
                "question": user_input,
                "answer": output,
                "timestamp": timestamp,
            }
        )
    except Exception as e:
        st.error(f"An error occurred while saving data to Airtable: {e}")

# Function to perform conversational chat
def conversational_chat(user_input):
    result = agent_executor({"input": user_input})
    st.session_state.chat_history.append((user_input, result["output"]))
    return result["output"]
    
output = ""
with container:
    if st.session_state.user_name is None:
        user_name = st.text_input("Your name:")
        if user_name:
            st.session_state.user_name = user_name

    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Type your question here (:")
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = conversational_chat(user_input)

    with response_container:
        for i, (query, answer) in enumerate(st.session_state.chat_history):
            user_name = st.session_state.user_name
            message(query, is_user=True, key=f"{i}_user", avatar_style="thumbs")
            col1, col2 = st.columns([0.7, 10]) 
            with col1:
                st.image("icon-1024.png", width=50)
            with col2:
                st.markdown(
                f'<div style="background-color: black; color: white; border-radius: 10px; padding: 10px; width: 60%;'
                f' border-top-right-radius: 10px; border-bottom-right-radius: 10px;'
                f' border-top-left-radius: 0; border-bottom-left-radius: 0; box-shadow: 2px 2px 5px #888888;">'
                f'<span style="font-family: Arial, sans-serif; font-size: 16px; white-space: pre-wrap;">{answer}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        if st.session_state.user_name:
            try:
                save_chat_to_airtable(st.session_state.user_name, user_input, output)
            except Exception as e:
                st.error(f"An error occurred: {e}")
