from fastapi import FastAPI, Request

app = FastAPI

@app.post('/webook')
async def webhook(request: Request):
    data = await request.json()
    print(data)
    return {'status': 'ok'}

