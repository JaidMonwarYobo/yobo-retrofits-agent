import ast
import json
import os
import re
import time
from io import BytesIO

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_types import AgentType
from langchain.chains import LLMChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from PIL import Image
from sqlalchemy import inspect, text
from streamlit_chat import message

# Initialize Streamlit app
st.set_page_config(page_title="SQL Agent Interface", layout="wide")
st.title("Yobo AI X Simply Retrofits: SQL Database Agent Interface")
chat_state = False

# Load environment variables
load_dotenv()

# Retrieve OpenAI API key from environment
# openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_key = st.secrets["OPENAI_API_KEY"]

if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in your `.env` file.")
    st.stop()


# Function to query as a list
def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))


# Function to initialize the agent
@st.cache_resource
def initialize_agent(
    db_uri,
    primary_table,
    product_name_col,
    secondary_tables,
    additional_info_columns,
    primary_foreign_keys,
    secondary_foreign_keys,
):
    # Initialize the database
    db = SQLDatabase.from_uri(db_uri)

    st.session_state["table_config"] = {
        "primary_table": primary_table,
        "product_name_col": product_name_col,
        "secondary_tables": secondary_tables,
        "additional_info_columns": additional_info_columns,
        "primary_foreign_keys": primary_foreign_keys,
        "secondary_foreign_keys": secondary_foreign_keys,
    }

    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)

    # Define examples for semantic similarity
    examples = [
        {"input": "List all artists.", "query": "SELECT * FROM Artist;"},
        {
            "input": "Find all albums for the artist 'AC/DC'.",
            "query": "SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'AC/DC');",
        },
    ]

    # Initialize example selector
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(model="text-embedding-ada-002"),
        FAISS,
        k=10,
        input_keys=["input"],
    )

    # Modified System Prompt
    system_prefix = """You are an agent designed to find and output proper names from a vector database.
When given an input, you must return only the proper name found in the vector database.
Do not perform any SQL operations or generate any SQL queries.
Do not provide any additional analysis, commentary, or other information.
Only return the name from the vector database."""

    # Create a simple retriever-only prompt
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template(
            "User input: {input}\nRetrieved name: {query}"
        ),
        input_variables=["input"],
        prefix=system_prefix,
        suffix="",
    )

    # Create the full chat prompt
    full_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(prompt=few_shot_prompt),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # Define the retriever tool with vector DB
    query_str = f"SELECT {product_name_col} FROM {primary_table}"
    product_name_list = query_as_list(db, query_str)
    vector_db = FAISS.from_texts(
        product_name_list, OpenAIEmbeddings(model="text-embedding-ada-002")
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 10})
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_proper_nouns",
        description="Use to look up proper nouns. Input is an approximate spelling of the proper noun, output is valid proper nouns.",
    )

    # Create the SQL agent
    agent = create_sql_agent(
        llm=llm,
        db=db,
        extra_tools=[retriever_tool],
        prompt=full_prompt,
        agent_type="openai-tools",
        verbose=False,
    )

    return agent, db, str(db.dialect), db.get_usable_table_names()


# Function to display image from URL
def display_image(url):
    try:
        with st.spinner(f"Loading image from {url}..."):
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                raise ValueError(
                    f"URL does not point to an image (content-type: {content_type})"
                )
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="Post Image", use_column_width=True)
            return True
    except Exception as e:
        st.error(f"Failed to load image: {str(e)}")
        st.write(f"Image URL: {url}")
        return False


def response_after_executing_sql_query(product_name, keywords):
    db = st.session_state["db"]
    engine = db._engine  # Get the SQLAlchemy engine from the SQLDatabase object
    config = st.session_state["table_config"]

    # Build the SQL query dynamically
    base_query = f"SELECT {config['primary_table']}.*"
    for i, additional_col in enumerate(config["additional_info_columns"]):
        base_query += f", {config['secondary_tables'][i]}.{additional_col}"

    # Add FROM and JOIN clauses
    base_query += f" FROM {config['primary_table']}"
    for i in range(len(config["secondary_tables"])):
        base_query += f" LEFT JOIN {config['secondary_tables'][i]} ON {config['primary_table']}.{config['primary_foreign_keys'][i]} = {config['secondary_tables'][i]}.{config['secondary_foreign_keys'][i]}"

    # Initialize where_conditions list
    where_conditions = []
    params = {}
    param_counter = 0  # Counter for unique parameter names

    # Add conditions for primary table's product_name_col
    for keyword in keywords:
        param_name = f"kw{param_counter}"
        where_conditions.append(
            f"LOWER({config['primary_table']}.{config['product_name_col']}) LIKE LOWER(:{param_name})"
        )
        params[param_name] = f"%{keyword}%"
        param_counter += 1

    # Add conditions for each search column
    search_columns = st.session_state["search_columns"]
    for search_col in search_columns:
        if search_col:  # Only add condition if search_col is not None
            for keyword in keywords:
                param_name = f"kw{param_counter}"
                where_conditions.append(
                    f"LOWER({config['primary_table']}.{search_col}) LIKE LOWER(:{param_name})"
                )
                params[param_name] = f"%{keyword}%"
                param_counter += 1

    # Add conditions for each additional column
    for i, additional_col in enumerate(config["additional_info_columns"]):
        for keyword in keywords:
            param_name = f"kw{param_counter}"
            where_conditions.append(
                f"LOWER({config['secondary_tables'][i]}.{additional_col}) LIKE LOWER(:{param_name})"
            )
            params[param_name] = f"%{keyword}%"
            param_counter += 1

    # Combine conditions with OR
    where_clause = " WHERE " + " OR ".join(where_conditions)

    # Complete the query string
    query_str = base_query + where_clause

    # Define the SQL query with parameter substitution
    query = text(query_str)

    # Execute the query
    with engine.connect() as connection:
        result = connection.execute(query, params)
        rows = result.fetchall()
        columns = result.keys()

    # Return the results as a list of dictionaries
    result_list = [dict(zip(columns, row)) for row in rows]
    return result_list


def extract_keywords(user_input):
    # Define a prompt template that instructs the LLM to extract keywords
    prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "Extract the product-related keywords from the following text. "
            "Output the keywords as a comma-separated list:\n\n"
            "{text}\n\nKeywords:"
        ),
    )
    # Initialize the ChatOpenAI model with a low temperature for deterministic output
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    # Create an LLMChain with the prompt and the model
    chain = LLMChain(llm=llm, prompt=prompt)
    # Run the chain to get the keywords
    keywords_text = chain.run(user_input)
    # Split the output into a list of keywords
    keywords_list = [keyword.strip() for keyword in keywords_text.split(",")]
    return keywords_list


def final_agent_response(product_df_results, user_input):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)

    chat_history = ChatMessageHistory()
    for pair in st.session_state.conversation_history:
        user_message = pair["user_message"]
        assistant_response = pair["assistant_response"]
        chat_history.add_message(HumanMessage(content=user_message))
        chat_history.add_message(AIMessage(content=assistant_response))

    # Create a memory object with the chat history
    memory = ConversationBufferMemory(
        chat_memory=chat_history, memory_key="chat_history", return_messages=True
    )

    # Create the agent with the memory
    agent = create_pandas_dataframe_agent(
        llm,
        product_df_results,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
        memory=memory,
        allow_dangerous_code=True,
    )

    return agent.run(user_input)


# Database credentials hardcoded
db_type = "MySQL"
host = "198.12.241.155"
port = "3306"
username = "inventory_yobo"
# password = os.getenv("DB_PASSWORD")  # Fetch password from environment variable
password = st.secrets["DB_PASSWORD"]  # Fetch password from environment variable
database = "inventory_yobo"

primary_table = "wp_data"
product_name_col = "wp_title"
search_columns = ["title", "short_des"]
secondary_tables = ["variations_data", "wp_product_categories"]
additional_info_columns = ["variation_name", "name"]
primary_foreign_keys = ["wp_id", "category"]
secondary_foreign_keys = ["wp_id", "categories_id"]

# Sidebar: Connect to Database Button
st.sidebar.header("Step 1: Database Connection")
connect_button = st.sidebar.button("Connect to Database")

if connect_button:
    # Construct the database URI
    db_uri = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"

    try:
        db = SQLDatabase.from_uri(db_uri)
        db_dialect = db.dialect
        table_names = db.get_usable_table_names()
        st.session_state["db"] = db
        st.session_state["db_uri"] = db_uri
        st.session_state["db_dialect"] = db_dialect
        st.session_state["table_names"] = table_names

        # Retrieve columns for each table
        inspector = inspect(db._engine)
        table_columns = {}
        for table in table_names:
            columns = inspector.get_columns(table)
            table_columns[table] = [column["name"] for column in columns]
        st.session_state["table_columns"] = table_columns

        st.success(f"Connected to the {db_type} database successfully!")

        # Hardcode schema configuration
        primary_table = "wp_data"
        product_name_col = "wp_title"
        search_columns = ["title", "short_des"]
        secondary_tables = ["variations_data", "wp_product_categories"]
        additional_info_columns = ["variation_name", "name"]
        primary_foreign_keys = ["wp_id", "category"]
        secondary_foreign_keys = ["wp_id", "categories_id"]

        # Store the search columns in session state
        st.session_state["search_columns"] = search_columns

    except Exception as e:
        st.error(f"Failed to connect to the database: {e}")
        st.stop()

st.sidebar.markdown("---")

# Keep only the Initialize Agent and Clear Chat buttons on the sidebar
st.sidebar.header("Step 2: Initialize Agent")
if st.sidebar.button("Initialize Agent"):
    try:
        # Use the hardcoded variables
        agent, db, db_dialect, table_names_updated = initialize_agent(
            st.session_state["db_uri"],
            primary_table,
            product_name_col,
            secondary_tables,
            additional_info_columns,
            primary_foreign_keys,
            secondary_foreign_keys,
        )

        # Assign to session_state
        st.session_state["agent"] = agent
        st.session_state["db"] = db
        st.session_state["db_dialect"] = db_dialect
        st.session_state["table_names"] = table_names_updated
        st.success("Product Query Agent initialized successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to initialize the agent: {e}")
        st.stop()

st.sidebar.markdown("---")
st.sidebar.header("Start a New Conversation Session")
if st.sidebar.button("Clear Chat"):
    st.session_state.chat_display = []
    st.session_state.conversation_history = []
    st.session_state.final_output = None

# Ensure the agent is initialized
if "agent" in st.session_state:
    agent = st.session_state["agent"]
    db_dialect = st.session_state["db_dialect"]
    table_names = st.session_state["table_names"]
else:
    st.warning("Please initialize the agent in the sidebar.")
    st.stop()

if "chat_display" not in st.session_state:
    st.session_state["chat_display"] = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "final_output" not in st.session_state:
    st.session_state.final_output = None

if "product_df_results" not in st.session_state:
    st.session_state.product_df_results = None

# Sidebar Information
st.sidebar.header("Database Information")
st.sidebar.write(f"**Dialect:** {db_dialect}")
st.sidebar.write(f"**Tables:** {', '.join(table_names)}")

# User Input
st.header("Enter Your Query")
user_input = st.text_input("Type your product-related question here:", "")


def new_product(conversation_history, user_input):
    system_prompt = f"""
You are an assistant that determines if the user's new question is about a different product than previously discussed.
Conversation history: {conversation_history}
If the user's new question is about the same product as in the conversation history, respond with 'no'.
If it is about a different product, or if it is a general statement (e.g., greetings, thanks, bye), respond with 'yes'.
Provide only 'yes' or 'no' as your response.
"""

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]

    deciding_agent = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=3,
        api_key=openai_api_key,
    )
    assistant_reply = deciding_agent(messages).content.strip()
    return assistant_reply


if st.button("Submit"):
    new_product_decider = new_product(st.session_state.conversation_history, user_input)
    if user_input.strip() == "":
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Processing your query..."):
            try:
                start_time = time.perf_counter()

                if new_product_decider.lower() == "yes":
                    # Agent processes the user input
                    response = agent.invoke({"input": user_input})

                    st.success("Query executed successfully!")

                    end_time_name = time.perf_counter()
                    elapsed_time_name_ms = (end_time_name - start_time) * 1000

                    if isinstance(response, dict) and "output" in response:
                        output = response["output"]
                    else:
                        output = str(response)

                    keywords = extract_keywords(user_input)
                    # Display the keywords (for debugging)
                    print("Keywords:", keywords)

                    # Get the final output from the database
                    final_output = response_after_executing_sql_query(output, keywords)
                    st.session_state.final_output = final_output

                    if final_output:
                        # Convert the list of dictionaries to a Pandas DataFrame
                        product_df_results = pd.DataFrame(final_output)
                        st.session_state.product_df_results = product_df_results

                        # Display the DataFrame in Streamlit
                        st.dataframe(product_df_results)
                    else:
                        st.write("No results found for the given keywords.")

                    natural_output = final_agent_response(
                        st.session_state.product_df_results, user_input
                    )

                    # Display the query results
                    if natural_output:
                        st.write("Yobo's Response:")
                        st.write(natural_output)
                    else:
                        st.write("No results found for the given product name.")

                    end_formatting_time = time.perf_counter()
                    elapsed_time_formatting_ms = (
                        end_formatting_time - end_time_name
                    ) * 1000

                    # Update conversation history
                    conversation_pair = {
                        "user_message": user_input,
                        "assistant_response": natural_output,
                    }

                    st.session_state.conversation_history.append(conversation_pair)

                    st.session_state.chat_display.append(
                        {"message": user_input, "is_user": True}
                    )
                    st.session_state.chat_display.append(
                        {"message": natural_output, "is_user": False}
                    )

                    st.markdown("---")
                    st.subheader("Conversation History")
                    st.markdown("---")
                    for i, chat in enumerate(st.session_state.chat_display):
                        message(chat["message"], is_user=chat["is_user"], key=str(i))

                    chat_state = True
                elif new_product_decider.lower() == "no":
                    final_output = st.session_state.final_output

                    if final_output:
                        # Display the DataFrame in Streamlit
                        st.dataframe(st.session_state.product_df_results)
                    else:
                        st.write("No results found for the given keywords.")

                    natural_output = final_agent_response(
                        st.session_state.product_df_results, user_input
                    )

                    st.write("Yobo's Response:")
                    st.write(natural_output)

                    elapsed_time_name_ms = 0
                    end_formatting_time = time.perf_counter()
                    elapsed_time_formatting_ms = (
                        end_formatting_time - start_time
                    ) * 1000

                    conversation_pair = {
                        "user_message": user_input,
                        "assistant_response": natural_output,
                    }

                    st.session_state.conversation_history.append(conversation_pair)

                    st.session_state.chat_display.append(
                        {"message": user_input, "is_user": True}
                    )
                    st.session_state.chat_display.append(
                        {"message": natural_output, "is_user": False}
                    )

                    st.markdown("---")
                    st.subheader("Conversation History")
                    st.markdown("---")
                    for i, chat in enumerate(st.session_state.chat_display):
                        message(chat["message"], is_user=chat["is_user"], key=str(i))

                    chat_state = True

                end_time = time.perf_counter()
                elapsed_time_ms = (end_time - start_time) * 1000

                st.info(
                    f"Time taken for Finding Name (t_1): {elapsed_time_name_ms:.2f} ms"
                )
                st.info(
                    f"Time taken for Executing Query and Formatting Query (t_2): {elapsed_time_formatting_ms:.2f} ms"
                )
                st.info(f"Total Response Time (t_1 + t_2): {elapsed_time_ms:.2f} ms")

            except Exception as e:
                st.error(f"An error occurred: {e}")

# Agent Logs (Optional)
st.markdown("---")
st.subheader("Agent Logs")
