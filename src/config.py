APP_NAME = "Smart Waste Sorter"

CATEGORIES = ["Plastic", "Paper", "Can_Metal", "Glass", "Organic"]
OTHER_CATEGORY = "Other"

DISPLAY_NAMES = {
    "Can_Metal": "Can / Metal",
}

DEFAULT_CONFIDENCE_THRESHOLD = 0.60
DEFAULT_REQUIRED_MATCHES = 3
POST_CONFIRMATION_PAUSE_SECONDS = 0.8
DEFAULT_VOICE_ENABLED = True
DEFAULT_VOICE_INCLUDE_CONFIDENCE = True
DB_PATH = "data/smart_waste.db"
MODEL_PATH = "model/best.pt"

BIN_RULES = {
    "Plastic": "Yellow plastic bin / Check local recycling acceptance",
    "Paper": "Blue paper recycling bin / Recyclable when clean and dry",
    "Can_Metal": "Blue metal recycling bin / Generally recyclable",
    "Glass": "Green glass recycling bin / Check local recycling acceptance",
    "Organic": "Green organic bin / Organic or compostable",
    "UNKNOWN": "Unknown / check the material and disposal guidance",
}

WASTE_INFO = {
    "Paper": {
        "display_name": "Paper",
        "recyclable": None,
        "recyclable_text": "Paper can be recycled when it is clean and dry",
        "bin_name": "paper recycling bin",
        "bin_color": "blue",
        "note": "Please check whether your local service accepts it",
    },
    "Cardboard": {
        "display_name": "Cardboard",
        "recyclable": None,
        "recyclable_text": "Unknown recycling status",
        "bin_name": "paper recycling bin",
        "bin_color": "blue",
        "note": (
            "Cardboard is generally recyclable when it is clean and dry. "
            "Please check local acceptance and flatten it before recycling"
        ),
    },
    "Plastic": {
        "display_name": "Plastic",
        "recyclable": None,
        "recyclable_text": "Some plastics can be recycled",
        "bin_name": "yellow plastic bin",
        "bin_color": "yellow",
        "note": "Please check whether your local recycling service accepts it",
    },
    "Glass": {
        "display_name": "Glass",
        "recyclable": None,
        "recyclable_text": "Glass may be recyclable",
        "bin_name": "green glass recycling bin",
        "bin_color": "green",
        "note": (
            "Please check whether your local service accepts it and handle "
            "broken glass carefully"
        ),
    },
    "Can_Metal": {
        "display_name": "Metal can",
        "recyclable": True,
        "recyclable_text": "Metal cans are generally recyclable",
        "bin_name": "metal recycling bin",
        "bin_color": "blue",
        "note": (
            "Please empty the can before recycling and check local acceptance"
        ),
    },
    "Organic": {
        "display_name": "Organic waste",
        "recyclable": False,
        "recyclable_text": (
            "Organic waste can be composted when suitable facilities are available"
        ),
        "bin_name": "organic waste bin",
        "bin_color": "green",
        "note": "",
    },
    "Food Waste": {
        "display_name": "Organic waste",
        "recyclable": False,
        "recyclable_text": "Organic or compostable",
        "bin_name": "organic waste bin",
        "bin_color": "green",
        "note": (
            "Organic waste can be composted when suitable facilities are available"
        ),
    },
    "Other": {
        "display_name": "Other waste",
        "recyclable": None,
        "recyclable_text": "Unknown recycling status",
        "bin_name": "general waste bin",
        "bin_color": "black",
        "note": (
            "Please check the material before disposing of it. This advice "
            "applies only when the Other category contains waste items"
        ),
    },
    "General Trash": {
        "display_name": "Other waste",
        "recyclable": None,
        "recyclable_text": "Unknown recycling status",
        "bin_name": "general waste bin",
        "bin_color": "black",
        "note": "Please check the material before disposing of it",
    },
    "Battery": {
        "display_name": "Battery",
        "recyclable": False,
        "recyclable_text": "Not recyclable in a normal bin",
        "bin_name": "hazardous waste bin",
        "bin_color": "red",
        "note": "Please take it to a battery collection point",
    },
    "E-Waste": {
        "display_name": "Electronic waste",
        "recyclable": False,
        "recyclable_text": "Not recyclable in a normal bin",
        "bin_name": "hazardous waste bin",
        "bin_color": "red",
        "note": "Please take it to an e-waste collection point",
    },
}

VOICE_DETECTION_TEMPLATE = "{name} detected. {confidence_text}"
VOICE_GUIDANCE_TEMPLATE = (
    "Please place this item in the {bin_name}. {recyclable_text}.{note_text}"
)
VOICE_UNKNOWN_MESSAGE = (
    "I am not sure what this item is. Please check it manually."
)
VOICE_NOT_RECOGNIZED_MESSAGE = (
    "This item is not recognized. Please check the correct disposal method."
)
VOICE_UNMAPPED_TEMPLATE = (
    "Item detected: {name}. No bin recommendation available."
)
VOICE_ERROR_TEMPLATE = "{message}"
VOICE_CONFIDENCE_TEMPLATE = "Confidence, {percent} percent."
VOICE_TEST_MESSAGE = (
    "Paper detected. Please put it in the paper recycling bin. Paper can be "
    "recycled when it is clean and dry. Please check whether your local "
    "service accepts it."
)
DEFAULT_VOICE_COOLDOWN_SECONDS = 2.0

STATE_FLOW = [
    "START",
    "READY",
    "DETECTING",
    "VALIDATE",
    "CONFIRMED",
    "LOCKED",
    "UNKNOWN",
    "ERROR",
]
