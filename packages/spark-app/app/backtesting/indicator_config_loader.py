import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from ..indicators.indicator_factory import IndicatorFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IndicatorConfigLoader:
    """
    Loads indicator configurations from YAML/JSON files and creates indicator instances.

    This class provides methods to load indicator configurations from files,
    validate their parameters, and create indicator instances using the IndicatorFactory.
    """

    @staticmethod
    def load_from_file(config_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load indicator configurations from a YAML/JSON file.

        Args:
            config_path: Path to the configuration file

        Returns:
            List of indicator configurations

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the file format is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        file_ext = config_path.suffix.lower()

        try:
            with open(config_path, "r") as file:
                if file_ext == ".json":
                    config_data = json.load(file)
                elif file_ext in [".yaml", ".yml"]:
                    config_data = yaml.safe_load(file)
                else:
                    raise ValueError(f"Unsupported file format: {file_ext}")

            # Validate that we have a list of indicator configurations
            if not isinstance(config_data, list):
                # Check if it might be a dictionary with 'indicators' as a list
                if isinstance(config_data, dict) and "indicators" in config_data and isinstance(config_data["indicators"], list):
                    config_data = config_data["indicators"]
                else:
                    raise ValueError("Configuration file must contain a list of indicator configurations or a dictionary with 'indicators' key")

            logger.info(f"Loaded {len(config_data)} indicator configurations from {config_path}")
            return config_data

        except (json.JSONDecodeError, yaml.YAMLError) as e:
            logger.error(f"Error parsing configuration file {config_path}: {e}")
            raise ValueError(f"Invalid file format in {config_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            raise

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Validate an indicator configuration.

        Args:
            config: Indicator configuration dictionary

        Returns:
            True if configuration is valid, False otherwise
        """
        required_fields = ["name", "type"]

        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field '{field}' in indicator configuration")
                return False

        # Check if the indicator type is supported
        if config["type"] not in IndicatorFactory.get_available_indicators():
            logger.error(f"Unsupported indicator type: {config['type']}")
            return False

        return True

    @classmethod
    def load_indicators(cls, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configurations from file and create indicator instances.

        Args:
            config_path: Path to the configuration file

        Returns:
            Dictionary of indicator instances keyed by name
        """
        configs = cls.load_from_file(config_path)

        # Validate configurations
        valid_configs = []
        for config in configs:
            if cls.validate_config(config):
                valid_configs.append(config)
            else:
                logger.warning(f"Skipping invalid indicator configuration: {config}")

        # Create indicators from valid configurations
        indicators = IndicatorFactory.create_indicators_from_config(valid_configs)

        logger.info(f"Created {len(indicators)} indicator instances from configuration")
        return indicators
