import asyncio
import logging
from quart import Quart
from quart_cors import cors


from config import async_read_indexes, init_env_and_logging, server_settings
from agent_workflow_structured_answer import builder #initialize a static workflow
from routes import register_routes

app = Quart(__name__)
app = cors(app, allow_origin="*")  # Allow CORS from all origins

init_env_and_logging()

# Register routes here
register_routes(app)


@app.before_serving
async def _spawn_loader_after_bind():
    # read indexes in a separate thread
    asyncio.create_task(async_read_indexes())
    logging.info("Scheduled index‐loader via create_task; server is live.")
    

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
