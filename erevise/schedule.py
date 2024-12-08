import logging
import requests
from datetime import datetime

current_datetime = datetime.now()

# Get an instance of a logger
logger = logging.getLogger('erevise')

def auto_click_button():
    try:
        # Simulate task logic. Replace with actual logic.
        logger.info(f"{current_datetime} Start Processing Essays.")

        url = "http://127.0.0.1:8000/process/"
        # Add necessary data or headers depending on the request
        requests.post(url)

        # Assuming the task is successful
        logger.info(f"{current_datetime} Successfully Processed Essays.")
    except Exception as e:
        logger.error(f"{current_datetime} Error during processing: {e}")