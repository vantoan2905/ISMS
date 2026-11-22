from fastapi import WebSocket, WebSocketDisconnect


async def events_socket(websocket: WebSocket, service):
    await websocket.accept()
    queue = service.subscribe()
    try:
        while True:
            await websocket.send_json(await queue.get())
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        service.unsubscribe(queue)
