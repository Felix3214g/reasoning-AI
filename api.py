from flask import Flask, request, jsonify
from ki import NeoCortex
from functools import wraps
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
cortex = NeoCortex()

# API key for authentication
API_KEY = "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"

# Authentication decorator
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Check for API key in headers or query parameters
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not api_key or api_key != API_KEY:
            return jsonify({'error': 'Unauthorized: Invalid or missing API key'}), 401
        
        return f(*args, **kwargs)
    return decorated

@app.route('/api/solve', methods=['POST'])
@require_api_key
def solve():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': 'Missing query parameter'}), 400
    
    show_work = data.get('show_work', True)
    use_agents = data.get('use_agents', True)
    
    try:
        result = cortex.solve(data['query'], show_work, use_agents)
        return jsonify(result)
    except AttributeError as e:
        if 'context_window_manager' in str(e):
            # Handle the specific missing attribute error
            error_msg = "Missing 'context_window_manager' in NeoCortex. Please add this attribute to your NeoCortex class in ki.py."
            app.logger.error(f"API Error: {error_msg}")
            return jsonify({'error': error_msg}), 500
        # Other attribute errors
        app.logger.error(f"API AttributeError: {str(e)}")
        return jsonify({'error': f"AttributeError: {str(e)}"}), 500
    except Exception as e:
        app.logger.error(f"API Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/agents/status', methods=['GET'])
@require_api_key
def get_agent_status():
    if not hasattr(cortex, 'multi_agent_system'):
        return jsonify({'error': 'Multi-agent system not initialized'}), 404
    
    try:
        agent_status = cortex.multi_agent_system.get_agent_status()
        return jsonify(agent_status)
    except AttributeError as e:
        if 'context_window_manager' in str(e):
            # Handle the specific missing attribute error
            error_msg = "Missing 'context_window_manager' in NeoCortex. Please add this attribute to your NeoCortex class in ki.py."
            app.logger.error(f"API Error: {error_msg}")
            return jsonify({'error': error_msg}), 500
        # Other attribute errors
        app.logger.error(f"API AttributeError: {str(e)}")
        return jsonify({'error': f"AttributeError: {str(e)}"}), 500
    except Exception as e:
        app.logger.error(f"API Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/fast-response', methods=['POST'])
@require_api_key
def fast_response():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': 'Missing query parameter'}), 400
    
    show_work = data.get('show_work', True)
    
    try:
        result = cortex._fast_response(data['query'], show_work)
        return jsonify(result)
    except AttributeError as e:
        if 'context_window_manager' in str(e):
            # Handle the specific missing attribute error
            error_msg = "Missing 'context_window_manager' in NeoCortex. Please add this attribute to your NeoCortex class in ki.py."
            app.logger.error(f"API Error: {error_msg}")
            return jsonify({'error': error_msg}), 500
        # Other attribute errors
        app.logger.error(f"API AttributeError: {str(e)}")
        return jsonify({'error': f"AttributeError: {str(e)}"}), 500
    except Exception as e:
        app.logger.error(f"API Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 