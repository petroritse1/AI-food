import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# Initialize the Groq LLM
client = ChatGroq(temperature=0.4,groq_api_key="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",model_name="llama-3.1-70b-versatile")

def create_chat_chain(system_prompt):
    """Creates the LLMChain for the chatbot with memory."""
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    
    chain = LLMChain(llm=client, prompt=prompt, memory=memory)
    return chain

# Initialize Streamlit app
st.set_page_config(page_title="LangChain Chatbot", page_icon="🤖")
st.title("LangChain Chatbot")

# Initialize chat history and prompt list in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prompts" not in st.session_state:
    st.session_state.prompts = []

# System prompt for the chatbot
system_prompt_template = """
You are a helpful AI assistant. Respond to user questions in a helpful manner and you will only response when the user asks a question
and not before.
"""
# Initialize chain in session state
if 'chain' not in st.session_state:
    st.session_state.chain = create_chat_chain(system_prompt_template)


# Sidebar for Prompt History
with st.sidebar:
    st.header("Prompt History")
    for i, prompt in enumerate(st.session_state.prompts):
            st.markdown(f'<div style="background-color: #e6f7ff; padding: 5px; border-radius: 5px;">{prompt}</div>', unsafe_allow_html=True)


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar = "🧑‍💻" if message['role']=='user' else "🤖"):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me anything"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Add prompt to the prompt history
    st.session_state.prompts.append(prompt)
    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Generate response from the LLM chain
    with st.chat_message("assistant", avatar = "🤖"):
        with st.spinner("Thinking..."):
            result = st.session_state.chain.run(prompt)
            st.markdown(result)
    
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": result})