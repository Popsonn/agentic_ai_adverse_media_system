# adverse_media_system/config/constants.py

from enum import Enum
# You reference AdverseMediaCategory but don't import it
# Should be:
from models.classification import AdverseMediaCategory  # or wherever it's defined

# List of credible news sources for search quality evaluation
CREDIBLE_SOURCES = [
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
    "cnn.com", "bbc.com", "ap.org", "forbes.com", "cnbc.com"
]

EXCLUDED_DOMAINS = [
    "linkedin.com", "facebook.com", "contactout.com", "twitter.com",
    "instagram.com", "tiktok.com", "youtube.com"
]
ADVERSE_KEYWORDS = {
    AdverseMediaCategory.FRAUD_FINANCIAL_CRIME: [
        "fraud", "money laundering", "tax evasion", "embezzlement", 
        "wire fraud", "ponzi scheme", "insider trading", "securities fraud"
    ],
    AdverseMediaCategory.CORRUPTION_BRIBERY: [
        "corruption", "bribery", "kickback", "political corruption", 
        "public corruption", "quid pro quo"
    ],
    AdverseMediaCategory.ORGANIZED_CRIME: [
        "drug trafficking", "arms smuggling", "human trafficking", 
        "organized crime", "criminal organization", "racketeering"
    ],
    AdverseMediaCategory.TERRORISM_EXTREMISM: [
        "terrorism", "terrorist financing", "extremism", "terror group", 
        "financing terrorism", "material support"
    ],
    AdverseMediaCategory.SANCTIONS_EVASION: [
        "sanctions evasion", "sanctions violation", "embargo violation", 
        "trade sanctions", "economic sanctions"
    
    ],
    AdverseMediaCategory.OTHER_SERIOUS_CRIMES: [
        "environmental crime", "illegal mining", "wildlife trafficking", 
        "cybercrime", "identity theft"
    ]
}

# Default search parameters
DEFAULT_SEARCH_PARAMS = {
    "topic": "general",
    "search_depth": "advanced",
    "extract_depth": "advanced",
    "format": "markdown"
}
