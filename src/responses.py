from src import personas
from asgiref.sync import sync_to_async

import openai
from dotenv import load_dotenv
import os

load_dotenv()
COMPLETIONS_MODEL = os.getenv('COMPLETIONS_MODEL')
INDEX_NAME = os.getenv('INDEX_NAME')

from src.embedding.database import get_redis_connection,get_redis_results

async def pre_handle(user_message) -> str:
    question_extract = openai.Completion.create(model=COMPLETIONS_MODEL,prompt=f"Extract the user's latest question from this message: {user_message}. Extract it and translate it in english as a sentence stating the Question")
    text = question_extract['choices'][0]['text']
    search_result = get_redis_results(get_redis_connection(),text,INDEX_NAME)['result'][0]
    result = "MORE OF THE WHITEPAPER\n" + search_result + "\n" + user_message
    print(result)
    return result

async def official_handle_response(message, client) -> str:
    return await sync_to_async(client.chatbot.ask)(message)

async def unofficial_handle_response(message, client) -> str:
    async for response in client.chatbot.ask(message):
        responseMessage = response["message"]
    return responseMessage

async def bard_handle_response(message, client) -> str:
    response = await sync_to_async(client.chatbot.ask)(message)
    responseMessage = response["content"]
    return responseMessage

async def bing_handle_response(message, client) -> str:
    async for response in client.chatbot.ask_stream(message):
        responseMessage = response
    if len(responseMessage[1]["item"]["messages"]) > 1 and "text" in responseMessage[1]["item"]["messages"][1]:
        return responseMessage[1]["item"]["messages"][1]["text"]
    else:
        await client.chatbot.reset()
        raise Exception("Bing is unable to continue the previous conversation and will automatically RESET this conversation.")

# prompt engineering
async def switch_persona(persona, client) -> None:
    if client.chat_model ==  "UNOFFICIAL":
        client.chatbot.reset_chat()
        async for _ in client.chatbot.ask(personas.PERSONAS.get(persona)):
            pass
    elif client.chat_model == "OFFICIAL":
        client.chatbot = client.get_chatbot_model(prompt=personas.PERSONAS.get(persona))
    elif client.chat_model == "Bard":
        client.chatbot = client.get_chatbot_model()
        await sync_to_async(client.chatbot.ask)(personas.PERSONAS.get(persona))
    elif client.chat_model == "Bing":
        await client.chatbot.reset()
        async for _ in client.chatbot.ask_stream(personas.PERSONAS.get(persona)):
            pass
