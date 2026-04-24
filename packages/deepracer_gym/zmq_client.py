import zmq
import msgpack

import msgpack_numpy as m
m.patch()


PORT: int = int(__import__('os').environ.get('GYM_PORT','8888'))
HOST: str = __import__('os').environ.get('GYM_HOST','127.0.0.1')
TIMEOUT_LONG: int=500_000   # ~8.3 m
TIMEOUT_SHORT: int=100_000  # ~1.7 m


class DeepracerClientZMQ:
    def __init__(self, host: str=HOST, port: int=PORT):
        self.host = host
        self.port = port
        self.socket = zmq.Context().socket(zmq.REQ)

        # Large timout for first connection
        self.socket.set(zmq.SNDTIMEO, TIMEOUT_LONG)
        self.socket.set(zmq.RCVTIMEO, TIMEOUT_LONG)

        self.socket.connect(f'tcp://{self.host}:{self.port}')
        import sys; print(f'[zmq_client] connecting tcp://{self.host}:{self.port}', flush=True, file=sys.stderr)
    
    def ready(self):
        message: dict[str, int] = {'ready': 1}
        self._send_message(message)

    def recieve_response(self):
        packed_response = self.socket.recv()
        response = msgpack.unpackb(packed_response)
        return response

    def send_message(self, message: dict[str, object]):
        self._send_message(message)
        response = self.recieve_response()
        return response
    
    def _send_message(self, message: dict[str, object]):
        packed_message = msgpack.packb(message)
        self.socket.send(packed_message)

    def __del__(self):
        self.socket.close()