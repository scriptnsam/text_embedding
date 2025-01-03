from flask import Flask,request,jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
model = None # Initialize model outside the route

@app.before_request
def load_model():
    global model
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        #Handle the error appropriately, perhaps exit the app
        import sys
        sys.exit(1)

@app.route('/embeddings', methods=['POST'])
def get_embeddings():
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" in request body'}), 400
        
        text = data['text']
        if isinstance(text,str):
            embeddings = model.encode(text).tolist()
            return jsonify({'embeddings': embeddings})
        elif isinstance(text, list):
            embeddings = [model.encode(t).tolist() for t in text]
            return jsonify({'embeddings': embeddings})
        else:
            return jsonify({'error': 'Invalid "text" format. Must be string or list of strings.'}), 400
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': 'Error generating embeddings'}), 500    
    
if __name__ == '__main__':
    # app.run(debug=True,port=5001) #Run on port 5001 (or any available port)
    app.run()