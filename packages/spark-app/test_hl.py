import logging

import requests
from hyperliquid.info import Info
from hyperliquid.utils import constants

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test addresses
TEST_ADDRESS = "0x0000000000000000000000000000000000000000"

def main():
    """Test different Hyperliquid API endpoints and connections"""
    api_base_url = "https://api.hyperliquid.xyz"

    logger.info(f"Testing Hyperliquid API at {api_base_url}")

    # Test direct API endpoints using requests
    logger.info("\n=== Testing direct API endpoints ===")
    endpoints = [
        "",  # base URL
        "/info",
        "/spot/balances",
        "/vault/user-state",
        "/info/user",
        "/info/meta",
    ]

    for endpoint in endpoints:
        url = f"{api_base_url}{endpoint}"
        try:
            response = requests.get(url, timeout=10)
            logger.info(f"Endpoint {url}: Status {response.status_code}")
            if response.status_code == 200:
                logger.info(f"Response: {response.text[:100]}...")
        except Exception as e:
            logger.error(f"Error accessing {url}: {e}")

    # Test SDK endpoints
    logger.info("\n=== Testing Hyperliquid SDK endpoints ===")
    info = Info(base_url=api_base_url)

    # Test meta endpoint
    try:
        meta = info.meta()
        logger.info(f"Meta endpoint works: {type(meta)}")
        logger.info(f"Found {len(meta.get('universe', []))} markets")
    except Exception as e:
        logger.error(f"Meta endpoint error: {e}")

    # Test user_state endpoint with a test address
    try:
        user_state = info.user_state(TEST_ADDRESS)
        logger.info(f"User state endpoint works: {type(user_state)}")
        logger.info(f"User cash: {user_state.get('cash', 'N/A')}")
    except Exception as e:
        logger.error(f"User state endpoint error: {e}")

    # Test all_mids endpoint
    try:
        mids = info.all_mids()
        logger.info(f"All mids endpoint works: {type(mids)}")
        logger.info(f"Found {len(mids)} price points")
    except Exception as e:
        logger.error(f"All mids endpoint error: {e}")

    # Check API version
    try:
        logger.info(f"API version from constants: {constants.API_VERSION}")
        logger.info(f"API domain from constants: {constants.API_DOMAIN_MAIN}")
    except Exception as e:
        logger.error(f"Error accessing constants: {e}")

if __name__ == "__main__":
    main()
