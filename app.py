from fastapi import FastAPI, Request
from evolution_api import send_whatsapp_message
from chains import get_conversational_rag_chain
from memory import get_session_history

app = FastAPI()

convertional_rag_chain = get_conversational_rag_chain()

def clear_session_history(chat_id: str):
    """
    Clear the chat history for a specific chat ID.
    This preserves the vectorstore while removing all conversation history.
    """
    try:
        # Get the session history for the chat ID
        session_history = get_session_history(chat_id)
        
        # Clear the messages
        session_history.clear()
        return True
    except Exception as e:
        print(f"Error clearing history: {str(e)}")
        return False

@app.post('/webhook')
async def webhook(request: Request):
    data = await request.json()
    print(data)
    chat_id = data.get('data').get('key').get('remoteJid')
    message = data.get('data').get('message').get('conversation')

    if not (chat_id and message and not '@g.us' in chat_id) or not ('553493090525' in chat_id or '553491143442' in chat_id or '553492999993' in chat_id):
        return {'status': 'ok'}
    
    # Check if the message is to clear history
    if message.strip().lower() == "limpar histórico" or message.strip().lower() == "limpar historico":
        print(f"Clearing history for {chat_id}")
        clear_session_history(chat_id)
        send_whatsapp_message(
            number=chat_id,
            text="Histórico de conversa limpo. Como posso ajudar?",
        )
        return {'status': 'ok', 'message': 'History cleared'}
    
    # Process normal message
    print(f"Processing query: {message}")
    ai_response = convertional_rag_chain.invoke(
        input={'input': message},
        config={'configurable':{'session_id':chat_id}},
    ).get('answer')
    send_whatsapp_message(
        number=chat_id,
        text=ai_response,
    )
    return {'status': 'ok'}

@app.post('/clear-history/{chat_id}')
async def clear_history(chat_id: str):
    """
    Clear the chat history for a specific chat ID.
    This preserves the vectorstore while removing all conversation history.
    """
    try:
        # Get the session history for the chat ID
        session_history = get_session_history(chat_id)
        
        # Clear the messages
        session_history.clear()
        
        return {
            'status': 'success',
            'message': f'Chat history cleared for {chat_id}'
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Failed to clear chat history: {str(e)}'
        }
