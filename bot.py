import streamlit as st
from utils import write_message, get_session_id
from agent_lan import generate_response

st.set_page_config("Assistente de cidades inteligentes", page_icon="💡")

if "messages" not in st.session_state:
    st.session_state.messages = [ #mostra mensagens na memoria
        {"role": "assistant", "content": "Olá! Sou o assistente de cidades inteligentes. Como posso ajudar hoje?"},
    ]

# Submit handler
def handle_submit(message):
    with st.spinner('Pensando...'):
        session_id = get_session_id() # <--- CORRIGIDO
        response = generate_response(message, session_id) # <--- CORRIGIDO #chama a resposta
        write_message('assistant', response) #grava mensagens

# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if prompt := st.chat_input("Olá! Como posso ajudar?"): #recebe a entrada do usuario
    write_message('user', prompt)
    handle_submit(prompt)

