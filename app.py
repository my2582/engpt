import os
import streamlit as st
import pandas as pd
from pandas import DataFrame
from google.oauth2 import service_account
import gspread
import openai
import numpy as np
from transformers import GPT2TokenizerFast
from sentence_transformers import SentenceTransformer
import datetime
from streamlit_chat import message
import random


def check_password() -> bool:
    """
    Returns `True` if the user had a correct password.
    :return:
    """

    def password_entered() -> bool:
        """
        Checks whether a password entered by the user is correct.
        """
        if st.session_state['username'] in st.secrets['passwords'] and \
                st.session_state['password'] == st.secrets['passwords'][st.session_state['username']]:
            st.session_state['password_correct'] = True
            del st.session_state['password']  # Never store password
        else:
            st.session_state['password_state'] = False

    if 'password_correct' not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input('아이디(ID)', value='a', on_change=password_entered, key='username')
        st.text_input('비밀번호', value='a', type='password', on_change=password_entered, key='password')
        return False
    elif not st.session_state['password_correct']:
        st.text_input('아이디(ID)', on_change=password_entered, key='username')
        st.text_input('비밀번호', type='password', on_change=password_entered, key='password')
        st.error('알 수 없는 아이디(ID) 또는 비밀번호 오류입니다.')
        return False
    else:
        return True


# if check_password():
#     if 'kept_username' not in st.session_state:
#         st.session_state['kept_username'] = st.session_state['username']
st.session_state['kept_username'] = random.random()

st.set_page_config(page_title = "서툰 영어를 해도 문법교정은 물론 자연스러운 표현도 알려 줘요.")
st.title('EngChat with ChatGPT: 영어, 이제 ChatGPT에게 배우세요.')
st.subheader('지치지 않는 원어민 AI를 준비했어요.')
st.markdown('\nChatGPT에게 문법적 오류와 자연스러운 원어민 표현을 배워 보세요.')
st.markdown('입력 예) I like Clasic music to listen.')
st.markdown('ChatGPT 답')
st.markdown('Change it to standard English:')
st.markdown('- The sentence should be \"I like to listen to classical music.\"')
st.markdown('\nPoint out every grammar mistake:')
st.markdown('-There is no verb after \"I like\".')
st.markdown('The word \"Clasic\" should be spelled \"Classical\".')
st.markdown('\nParaphrase to make it sound more natural to English native speakers:')
st.markdown('I\'m a fan of classical music and enjoy listening to it.')
st.markdown('Translate all of your answers to Korean:')
st.markdown('- 나는 클래식 음악을 들으려고 한다.')
st.markdown('- 빈칸에는 동사가 필요합니다.')
st.markdown('- 클래식은 클래시컬이라고 철자해야 합니다.')
st.markdown('- 클래식 음악을 들을 때 좋아한다는 것을 영어의 원어민으로 더 자연스럽게 말하면, \"나는 클래식 음악을 좋아하고 들으려고 한다\" 정도 라고 할 수 있습니다.')
st.markdown('\nChatGPT에게 명령 내리기: 아래 버튼을 누르세요.')

method = 'openai'
openai.api_key = st.secrets['open_ai_key']

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'direct_instruction' not in st.session_state:
    st.session_state['direct_instruction'] = False

init_msg = ''


@st.cache_resource
def load_tokenizer():
    return GPT2TokenizerFast.from_pretrained('gpt2')

@st.cache_resource
def access_sheet(sheet_name: str):
    """
    Access the Google's sheets.
    :param sheet_name:
    :return:
    """

    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    credentials = service_account.Credentials.from_service_account_info(st.secrets['gcp_service_account'],
                                                                        scopes=scope)

    gc = gspread.authorize(credentials)
    # open by a title that appears on a Google Sheet.
    sheet = gc.open('engpt-db').worksheet(sheet_name)

    return sheet

def get_max_num_tokens() -> int:
    """
    The maximum number of tokens a pre-trained NLP model can take.
    :return:
    """

    return 2046

def construct_prompt(query: str,  method: str):
    '''
    Construct the prompt to answer the query. The prompt is composed of the query from the user.
    :param query: str
        Query.
    :param method:
        indicates which model to use.
    :return:
    '''

    MAX_SECTION_LEN = get_max_num_tokens()
    SEPARATOR = '\n* '
    tokenizer = load_tokenizer()
    separator_len = len(tokenizer.tokenize(SEPARATOR))
    chosen_sections = []
    chosen_sections_len = 0

    header = '''
You are an ESL teacher answering students' questions. Reply as an ESL teacher.

Example conversion:
    
Student: Hi!
    
Teacher: Hi there! How can I help you?
    
Student: I want to learn English from you. Would you help me?
    
Teacher: Hi! I would be happy to help you with your English language learning. What kind of help do you need?

Student: For the following text in double quotations, do: change it to standard English,
point out every grammar mistake, paraphrase to make it sound more natural to English native speakers.
Answer in bullet points. Translate all of your answers to Korean in a separate answer.'''

    prompt = f'{header}\n"{query}\"\n\nTeacher: '

    return prompt

def record_question_answer(user, query, answer):
    """
    record the query, prompt and answer in the database (google sheet).
    """

    sheet = access_sheet('Q&A')
    data = sheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=['user', 'date', 'query', 'answer'])
    num_records = len(df)
    today_str = datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d')
    sheet.update_cell(num_records+2, 1, user)
    sheet.update_cell(num_records+2, 2, today_str)
    sheet.update_cell(num_records+2, 3, query)
    sheet.update_cell(num_records+2, 4, answer)


def chat_with_chatgpt(query, method, direct_instruction: bool):
    """
    Use a pre-trained NLP method to answer a question. The function also records the query, the prompt, and the
    answer in the database.
    :param query: str
        Query
    :param method:  str
        Method indicates which model to use, either 'openai' for using the OpenAPI for 'text-embedding-ada-002' or
        'huggingface' for using locally 'paraphrase-MiniLM-L6-v2'. In the former case, the output is only a string
        that will be used via the API. in the latter case, it is an actual model object.
    :return:
    answer: str
        Answer from the model.
    prompt: str
        Actual prompt built.
    """

    if st.session_state['direct_instruction']:
        prompt = st.session_state['direct_msg']
        st.session_state['direct_instruction'] = False
    else:
        prompt = construct_prompt(query, method)

    COMPLETIONS_MODEL = 'text-davinci-003'

    response = openai.Completion.create(
        prompt=prompt,
        temperature=0.9,
        max_tokens=800,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model=COMPLETIONS_MODEL
    )

    answer = response['choices'][0]['text'].strip(' \n')

    return answer, prompt


def on_clear_input_text():
    if str(st.session_state.input_widget):
        st.session_state.input_widget = ''


def on_ask_me_question():
    st.session_state['direct_instruction'] = True
    st.session_state['direct_msg'] = 'Ask me any casual question, which may encourage me to continue casual ' \
                                     'conversation with you.'


def get_text(init_msg):
    input_text = st.text_input('입력란:', init_msg, key='input_widget')
    return input_text


st.button('나한테 아무 질문이나 해 줘.', on_click=on_ask_me_question)
st.button('입력란 지워 줘.', on_click=on_clear_input_text)

user_input = get_text(init_msg=init_msg)
user_input = user_input.strip()

if st.session_state['direct_instruction']:
    user_input = st.session_state['direct_msg']

if user_input and user_input != '':
    answer, prompt = chat_with_chatgpt(query=user_input, method=method, direct_instruction=st.session_state['direct_instruction'] )

    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer)

    if user_input != init_msg:
        record_question_answer(st.session_state['kept_username'], user_input, answer)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['generated'][i], key=str(i), avatar_style='bottts')
        message(st.session_state['past'][i], is_user=True, key=str(i)+'_user', avatar_style='avataaars')

