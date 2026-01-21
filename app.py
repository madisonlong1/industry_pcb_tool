import os
import base64
import json
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv


# --- 1. CONFIGURATION ---
# Ensure your API key is set in your environment variables
# os.environ["OPENAI_API_KEY"] = "sk-..." 

from_code = "en"
to_code = "es"

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


client = OpenAI()

# The model identifier you requested. 
# NOTE: If "gpt-5.2" is not yet available to your account, fallback to "gpt-4o".
MODEL_ID = "gpt-5.2"

# --- 2. DEFINE THE OUTPUT SCHEMA (Structured Outputs) ---
# This strictly enforces the return of bounding boxes and categories.
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "pcb_analysis_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "object",
                    "properties": {
                        "Integrated Circuits": {"type": "integer"},
                        "Electrolytic Capacitors": {"type": "integer"},
                        "Tantalum Capacitors": {"type": "integer"},
                        "Large MLCCs": {"type": "integer"},
                        "Connector Blocks": {"type": "integer"}
                    },
                    "required": ["Integrated Circuits", "Electrolytic Capacitors", "Tantalum Capacitors", "Large MLCCs", "Connector Blocks"],
                    "additionalProperties": False
                },
                "detections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": ["Integrated Circuits", "Electrolytic Capacitors", "Tantalum Capacitors", "Large MLCCs", "Connector Blocks"]
                            },
                            "part_number": {
                                "type": "string",
                                "description": "Visible alphanumeric text. Return 'null' if unreadable."
                            },
                            "box_2d": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "The bounding box [ymin, xmin, ymax, xmax] normalized to 0-1."
                            }
                        },
                        "required": ["category", "part_number", "box_2d"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["summary", "detections"],
            "additionalProperties": False
        }
    }
}

# --- 3. SYSTEM PROMPT WITH DATA INJECTION ---
SYSTEM_PROMPT = """
You are ACI (Agentic Component Identifier). Your goal is to detect, localize, and identify electronic components on PCB images.

### USER DATA & VISUAL RULES:
(Use these rules to determine WHERE to look for part numbers)
1. **Integrated Circuits (ICs):** Dark rectangular packages. Part numbers are laser-etched on the top face (e.g., "ATMega328").
2. **Electrolytic Capacitors:** Cylindrical metal cans. Values (uF, Voltage) are printed on the top (aluminum cap) or the side casing.
3. **Tantalum Capacitors:** Usually yellow or black blocks. Text is printed on the top face.
4. **Large MLCCs:** Brown/beige rectangular blocks. These rarely have text. Mark part_number as "null" unless explicitly visible.
5. **Connector Blocks:** Plastic or metal interfaces. Look for stamped IDs on the housing.

### INSTRUCTIONS:
1. Detect all instances of the 5 target categories.
2. For each detection, provide a precise bounding box [ymin, xmin, ymax, xmax] (0-1 normalized).
3. Extract any visible part number or value text.
4. Return the result strictly in the defined JSON format.
"""

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_pcb', methods=['POST'])
def analyze_pcb():
    print(">>> Phase 1 (OpenAI) Analysis Started...")
    
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image data found"}), 400
        
        # Prepare the image for OpenAI (URL or Base64)
        # OpenAI expects a data URL for base64 images
        base64_image = data['image']
        mime_type = data.get('mime_type', 'image/jpeg')
        image_url = f"data:{mime_type};base64,{base64_image}"

        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Analyze this PCB."},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            response_format=SCHEMA, # Enforces the strict JSON schema
            temperature=0.1,
            max_completion_tokens=4096
        )
        print("Actual model used:", response.model)

        # Parse the Structured Output
        result_content = response.choices[0].message.content
        parsed_data = json.loads(result_content)
        
        print(f"Success. Detected {len(parsed_data.get('detections', []))} items.")
        return jsonify(parsed_data)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)