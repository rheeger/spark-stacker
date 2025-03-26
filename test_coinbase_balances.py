#!/usr/bin/env python3

import os
import logging
import json
from coinbase.rest import RESTClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    try:
        # Get API credentials from environment
        api_key = os.environ.get("COINBASE_API_KEY")
        api_secret = os.environ.get("COINBASE_API_SECRET")

        if not api_key or not api_secret:
            logger.error("Missing API credentials in environment variables")
            return

        logger.info(f"Creating RESTClient with API key: {api_key[:20]}...")
        client = RESTClient(api_key=api_key, api_secret=api_secret, verbose=True)

        # Test connection by getting accounts
        logger.info("Fetching accounts...")
        accounts_response = client.get_accounts()

        # Create a results dictionary to save to file
        results = {
            "response_type": str(type(accounts_response)),
            "response_attrs": [
                attr
                for attr in dir(accounts_response)
                if not attr.startswith("_")
                and not callable(getattr(accounts_response, attr))
            ],
            "accounts": [],
        }

        if hasattr(accounts_response, "accounts"):
            logger.info(f"Found {len(accounts_response.accounts)} accounts")
            results["account_count"] = len(accounts_response.accounts)

            # Check for accounts with balance
            non_zero_balances = 0
            for i, account in enumerate(
                accounts_response.accounts[:20]
            ):  # Show first 20 accounts
                acct_data = {"index": i}

                # Extract basic account info
                for attr in ["uuid", "name", "currency", "type", "active", "ready"]:
                    if hasattr(account, attr):
                        acct_data[attr] = getattr(account, attr)

                # Extract balance info
                if hasattr(account, "available_balance") and hasattr(
                    account.available_balance, "value"
                ):
                    balance_value = float(account.available_balance.value)
                    acct_data["available_balance"] = balance_value
                    acct_data["balance_currency"] = getattr(
                        account.available_balance, "currency", None
                    )

                    if balance_value > 0:
                        non_zero_balances += 1
                        acct_data["non_zero"] = True
                    else:
                        acct_data["non_zero"] = False

                # Also check hold balance
                if hasattr(account, "hold") and hasattr(account.hold, "value"):
                    hold_value = float(account.hold.value)
                    acct_data["hold_balance"] = hold_value

                results["accounts"].append(acct_data)

                # Log to console
                currency = getattr(account, "currency", "UNKNOWN")
                balance_str = "N/A"
                if hasattr(account, "available_balance") and hasattr(
                    account.available_balance, "value"
                ):
                    balance_str = account.available_balance.value

                logger.info(f"Account {i}: {currency} - Available: {balance_str}")

            results["non_zero_balance_count"] = non_zero_balances
            logger.info(f"Found {non_zero_balances} accounts with non-zero balances")
        else:
            logger.error("Unexpected response format: no 'accounts' attribute")
            if hasattr(accounts_response, "to_dict"):
                results["response_dict"] = accounts_response.to_dict()

        # Save results to file
        with open("coinbase_accounts_test.json", "w") as f:
            json.dump(results, f, indent=2)
            logger.info("Results saved to coinbase_accounts_test.json")

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
