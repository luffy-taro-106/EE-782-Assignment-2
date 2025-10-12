# websocket_server.py
import asyncio
import websockets
import json

# Set of connected clients
connected_clients = set()

async def handler(websocket):
    # Register client
    connected_clients.add(websocket)
    print(f"[ws server] Client connected. Total clients: {len(connected_clients)}")
    try:
        async for message in websocket:
            data = json.loads(message)
            print("Received from GuardAgent:", data)

            # Broadcast to all connected clients except sender
            if connected_clients:
                msg = json.dumps(data)
                await asyncio.gather(*[
                    client.send(msg) for client in connected_clients
                    if client != websocket
                ])
    except websockets.ConnectionClosed:
        print("[ws server] Client disconnected")
    finally:
        connected_clients.remove(websocket)
        print(f"[ws server] Client removed. Total clients: {len(connected_clients)}")

async def main():
    server = await websockets.serve(handler, "localhost", 8765)
    print("[ws server] Running on ws://localhost:8765")
    await server.wait_closed()

asyncio.run(main())
