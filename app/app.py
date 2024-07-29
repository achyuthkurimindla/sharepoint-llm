from openai import OpenAI
import pandas as pd

import streamlit as st
import webbrowser
import random
import time
try:
    import chromadb
    from chromadb.config import Settings
except Exception as e:
    st.error(f"Error importing chromadb: {e}")
    raise
from openai import OpenAI

import boto3

import json


from fastapi.responses import JSONResponse

client=OpenAI(base_url="http://localhost:1234/v1",api_key="not-needed")



def mistral7b(user_message,system_message):
    completion=client.chat.completions.create(
        model="local model",
        messages=[
            {"role":"system","content":system_message},
            {"role":"user","content":user_message}
        ],
        temperature=0.7,
    )
    return completion.choices[0].message.content


def build_llama2_prompt(messages):
    startPrompt = "<s>[INST] "

    endPrompt = " [/INST]"

    conversation = []

    for index, message in enumerate(messages):

        if message["role"] == "system" and index == 0:

            conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")

        elif message["role"] == "user":

            conversation.append(message["content"].strip())

        else:

            conversation.append(f" [/INST] {message['content'].strip()}</s><s>[INST] ")
 
    return startPrompt + "".join(conversation) + endPrompt

def mistral7b_sharepoint(user_message, system_message):
    print("--step1")
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    print("--step2")
    prompt = build_llama2_prompt(messages)
    print("--step3")
    payload = {
        "inputs": prompt,
        "parameters": {
            "do_sample": True,
            "top_p": 0.6,
            "temperature": 0.8,
            "top_k": 50,
            "max_new_tokens": 512,
            "repetition_penalty": 1.03,
            "stop": ["</s>"]
        }
    }
    print("--step4")
    # Initialize the boto3 client for SageMaker runtime
    client = boto3.client('sagemaker-runtime', region_name='us-east-1')
    print("--step5")
    # Make the request to the endpoint
    try:
        response = client.invoke_endpoint(
            EndpointName='huggingface-pytorch-tgi-inference-2024-07-21-14-03-33-903',
            ContentType='application/json',
            Body=json.dumps(payload)
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    print("--step6")
    # Read the response
    response_body = response['Body'].read().decode('utf-8')
    print("--step7")
    response_json = json.loads(response_body)
    generated_text = response_json[0]['generated_text'] if isinstance(response_json, list) and len(response_json) > 0 else None
    return generated_text

def check_credentials(email, password):
    # Replace this with actual credential checking logic
    return email == "user@example.com" and password == "password"

# Function to render the login screen
def login_screen():
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Submit"):
        if check_credentials(email, password):
            st.session_state.logged_in = True
            st.session_state.user_name = email
            st.experimental_rerun()
        else:
            st.error("Invalid email or password")

def chatbot_screen(user_name):

    
    chroma_client = chromadb.HttpClient(host='3.108.209.221',port=8001)
    def response_generator(results):
            for word in results.split():
                yield word + " "
                time.sleep(0.05)
    
        # user_name = 'sai@gmail.com'
    if "messages" not in st.session_state:
            st.session_state.messages = []
    def chat():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    def chat_response():
        with st.chat_message(st.session_state.messages[-1]["role"]):
            st.write_stream(response_generator(st.session_state.messages[-1]['content']))

    container_1 = st.container(height=300,border=0)
    container_2 = st.container()
 
    with container_2:
        with st.form('prompt',border=0):
            genre = st.radio(
            "",
            ["***Chat with Database***", "***Chat with Sharepoint Files***", "***Chat with S3-bucket Files***"],
            index=0,
            horizontal=True,
            key='genre_radio',
            )
            user_input = st.text_input('',placeholder='Ask')
            submitted = st.form_submit_button("Submit")
        

    if user_input and genre == '***Chat with Database***':
        collection = chroma_client.get_collection(name='sales_embeddings')
        results = collection.query(
            query_texts=[user_input],
            n_results=5,
            # where={"user":  user_name}
            where = {"user": {"$in": ["sai@gmail.com"]}}
        )
        retrieved_documents1=results.get('documents',[])
        print(retrieved_documents1)
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        with container_1:
            chat()
        # with st.chat_message("user"):
        #     st.markdown(user_input)

        # results= 'Chat with Database'
        # system_message='''You are a chat bot only give crisp answer to the user prompts'''
        system_message = '''Based on the context data and vector search results, provide answers that directly address the user's question. Ignore any irrelevant or unrelated vectors. Focus on using the most relevant and accurate information available from the vector search results.'''
        # answer= f"Context: {results} \n User Query: {user_input}"
        user_message = f"Based on the context provided: {retrieved_documents1}, answer the following question: {user_input}. Ensure the answer is derived from the given context."

        # chatbot_response=mistral7b(user_message,system_message)
        # with st.chat_message("assistant"):
        #     # response = st.write_stream(response_generator())
        #     response = st.write_stream(response_generator(results))
        st.session_state.messages.append({"role": "assistant", "content": retrieved_documents1})
        # print(st.session_state.messages)
        with container_1:
            chat_response()
           
 
    elif user_input and genre == "***Chat with Sharepoint Files***":

        def file_selector_1(user_input, user_id):
            try:
                chroma_client = chromadb.HttpClient(host='3.108.209.221', port=8001)
                sharepoint_collection = chroma_client.get_or_create_collection(name="sharepoint_data_v1")
                

                results = sharepoint_collection.query(
                    query_texts=user_input, 
                    n_results=15,
                )

                if not results['metadatas'] or len(results['metadatas'][0]) == 0:
                    print("No documents found for the given query.")
                    return None

                unique_list = []
                for i in results['metadatas'][0]:
                    if 'users' in i and user_id in i['users']:
                        if i['source'] not in unique_list:
                            unique_list.append(i['source'])

                if not unique_list:
                    print(f"No documents found for the given user ID: {user_id}")
                    return "No Data Found for this user"

                final_results = sharepoint_collection.query(
                    query_texts=user_input, 
                    n_results=15,
                    where={'source': {"$in": unique_list}}
                )
                

                print(final_results)
                
                retrieved_documents1=final_results.get('documents',[])
                retrieved_content = "\n".join(retrieved_documents1[0])

                return retrieved_content

            except Exception as e:
                print(f"An error occurred: {e}")
                return None


        system_message='''You are a chat bot only give crisp answer to the user prompts'''
        st.session_state.messages.append({"role": "user", "content": user_input})
        with container_1:
            chat()      
        user_id = 'c93bae3f-c014-4372-b09e-d12f6a8c6fc7'
        
        retrieved_documents = file_selector_1(user_input, user_id)
        


        print(retrieved_documents)
        # user_message = f"Based on the following information: {retrieved_documents}.Answer to this question: {user_input}?"
        
        user_message = f"Based on the context provided: {retrieved_documents}, answer the following question: {user_input}. Ensure the answer is derived from the given context."
 
        # response = mistral7b_sharepoint(user_message, system_message)
        # print(response)

        # lm_stdio_response = mistral7b(user_message, system_message)
        # print(lm_stdio_response)
        
        results= 'Chat with Sharepoint Files'

        st.session_state.messages.append({"role": "assistant", "content": retrieved_documents})
        with container_1:
            chat_response()
            # chat()

    elif user_input and genre == "***Chat with S3-bucket Files***":
        
        
                
        def file_selector(prompt):
            try:
                chroma_client = chromadb.HttpClient(host='3.108.209.221', port=8001)
                s3_collection = chroma_client.get_or_create_collection(name="s3_data")
                
                results = s3_collection.query(
                query_texts=prompt, 
                n_results=15
                )
                retrieved_documents1=results.get('documents',[])
                retrieved_content = "\n".join(retrieved_documents1[0])
                return retrieved_content
            
                # return results
            
                unique_list=[]
                for i in results['metadatas'][0]:
                    for key in i.keys():
                        if i[key] not in unique_list and type(i[key])==str:
                            unique_list.append(i[key])
                
                return unique_list
                
                retrieved_documents1=final_results.get('documents',[])
                retrieved_content = "\n".join(retrieved_documents1[0])
                return retrieved_content

            except Exception as e:
                print(f"An error occurred: {e}")
                return None            

        
        
        
        system_message='''You are a chat bot only give crisp answer to the user prompts'''
        st.session_state.messages.append({"role": "user", "content": user_input})
        with container_1:
            chat() 
        # with st.chat_message("user"):
        #     st.markdown(user_input)
        # unique_list=file_selector(user_input)
        
        
        # print(unique_list)
        
        
        
        
        retrieved_documents = file_selector(user_input)
        print(retrieved_documents)
        
        user_message = f"Based on the context provided: {retrieved_documents}, answer the following question: {user_input}. Ensure the answer is derived from the given context."
        # lm_stdio_response = mistral7b(user_message, system_message)
        # print(lm_stdio_response)
        
        results= 'Chat with S3 Files'
        st.session_state.messages.append({"role": "assistant", "content": retrieved_documents})
        with container_1:
            chat_response()
        
        # with st.chat_message("assistant"):
            # response = st.write_stream(response_generator())

    else:
        pass
    


def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        chatbot_screen(st.session_state.user_name)
    else:

        chatbot_screen('hi')

if __name__ == "__main__":    
    main()
